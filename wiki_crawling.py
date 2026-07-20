"""
Wikipedia 크롤링 모듈 — XTools / Wikimedia REST API 기반 데이터 수집.

외부 인터페이스 (pipeline.py, app.py 등에서 호출):
    check_seed(dataset)        → list[list]
    n_char_crawler(split_list, n, path) → pd.DataFrame
    wiki_info_crawl(split_input, path)  → None
"""

import os
import re
import sys
import time
import datetime
import itertools
import urllib.parse
from collections import deque

import requests
import pandas as pd
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm.auto import tqdm
from nltk.tokenize import word_tokenize

import warnings
warnings.filterwarnings(action="ignore")
pd.set_option("expand_frame_repr", False)

# Wikipedia API 정책 준수용 고정 User-Agent
APP_USER_AGENT = "WikiResearchBot/1.0 (academic research project; Python/requests)"

# ============================================================
# 1. 공통 인프라
# ============================================================

def _create_session() -> requests.Session:
    """Retry + User-Agent가 설정된 세션을 생성."""
    session = requests.Session()
    retry = Retry(
        total=2,
        backoff_factor=0.5,
        status_forcelist=[500, 502, 503, 504],
        respect_retry_after_header=True,
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update({"User-Agent": APP_USER_AGENT})
    return session


def _request_with_retry(session, url, label="", max_retries=5, **kwargs):
    """429 Rate Limit + 네트워크 오류를 자동으로 처리하는 요청 함수.

    네트워크 오류(DNS 실패, 연결 거부 등) 발생 시:
    세션 연결 풀을 초기화하고 대기 후 재시도 (사용자가 앱을 껐다 켠 것과 동일 효과).
    """
    timeout = kwargs.pop("timeout", 30)
    resp = None
    for attempt in range(max_retries):
        try:
            resp = session.get(url=url, verify=False, timeout=timeout, **kwargs)
        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                OSError) as e:
            if attempt < max_retries - 1:
                wait = min(10 * (attempt + 1), 60)
                print(f"  [Network] {label} — 연결 실패, {wait}초 후 재시도 ({attempt+1}/{max_retries})")
                time.sleep(wait)
                # 세션 연결 풀 초기화 → 새 TCP 연결 강제 (앱 재시작과 동일 효과)
                try:
                    for adapter in session.adapters.values():
                        adapter.close()
                except Exception:
                    pass
                continue
            print(f"  [Network] {label} — {max_retries}회 재시도 실패, 건너뜀")
            return None

        if resp.status_code != 429:
            return resp
        try:
            wait = int(resp.headers.get("Retry-After", 30))
        except (ValueError, TypeError):
            wait = 30
        # Retry-After가 5분 이상이면 즉시 포기 (rate limit 해제까지 너무 오래 걸림)
        if wait > 300:
            print(f"  [Rate Limit] {label} — Retry-After={wait}초 (너무 김), 건너뜀")
            return None
        wait = min(wait, 120)
        print(f"  [Rate Limit] {label} — {wait}초 대기 후 재시도... ({attempt+1}/{max_retries})")
        time.sleep(wait)
    return resp


def _encode_title(title: str) -> str:
    """위키 제목을 URL에 안전하게 인코딩. 대문자 변환 없이 원본 유지."""
    return urllib.parse.quote(title.replace(" ", "_"), safe="")


def log_error(true_title, node, ex):
    """에러를 error_log.txt에 기록."""
    error_file = "./error_log.txt"
    try:
        with open(error_file, "a", encoding="utf-8-sig") as f:
            f.write(f"{true_title}  ||  {node} || {ex}\n")
    except OSError:
        pass
    print(f"{true_title} 시드의 {node} 에서 에러 발생 : {ex}")


# ============================================================
# 1-b. 함수 전체 타임아웃 데코레이터
# ============================================================

import threading
import functools

class _FuncTimeoutError(Exception):
    pass

def _with_timeout(seconds):
    """함수 전체 실행 시간을 제한하는 데코레이터.
    제한 시간 초과 시 빈 DataFrame을 반환."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = [pd.DataFrame()]
            error = [None]
            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    error[0] = e
            thread = threading.Thread(target=target, daemon=True)
            thread.start()
            thread.join(timeout=seconds)
            if thread.is_alive():
                print(f"  [Timeout] {func.__name__} — {seconds}초 초과, 건너뜀")
                return pd.DataFrame()
            if error[0]:
                raise error[0]
            return result[0]
        return wrapper
    return decorator


# ============================================================
# 2. 4개 수집 함수 (edit, pageviews, info, link)
# ============================================================

@_with_timeout(120)
def crawl_edit_by_year(title: str, start_year: int = 2001, end_year: int = 2025) -> pd.DataFrame:
    """
    연도별 편집 횟수 수집 (MediaWiki Action API).
    en.wikipedia.org/w/api.php 사용 — wikimedia.org 대비 rate limit 훨씬 넉넉 (15 req/s).
    전체 revision timestamp를 페이지네이션으로 가져와서 연도별 집계.
    """
    year_counts = {str(y): 0 for y in range(start_year, end_year + 1)}
    page_title = title.replace(" ", "_")

    with _create_session() as session:
        api_url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "prop": "revisions",
            "titles": title,
            "rvprop": "timestamp",
            "rvlimit": "500",
            "rvdir": "newer",
            "rvstart": f"{start_year}-01-01T00:00:00Z",
            "rvend": f"{end_year + 1}-01-01T00:00:00Z",
            "format": "json",
            "formatversion": "2",
        }

        while True:
            try:
                resp = _request_with_retry(
                    session, api_url, label=f"edits({title})",
                    params=params,
                )
                if resp is None or resp.status_code != 200:
                    break
                data = resp.json()
                pages = data.get("query", {}).get("pages", [])
                for page in pages:
                    if page.get("title"):
                        page_title = page["title"].replace(" ", "_")
                    for rev in page.get("revisions", []):
                        year = rev.get("timestamp", "")[:4]
                        if year in year_counts:
                            year_counts[year] += 1

                if "continue" in data:
                    params["rvcontinue"] = data["continue"]["rvcontinue"]
                else:
                    break
            except Exception:
                break

            time.sleep(0.1)

    rows = [[page_title, cnt, yr] for yr, cnt in sorted(year_counts.items())]
    return pd.DataFrame(rows, columns=["title", "edits", "year"])


@_with_timeout(60)
def crawl_pageviews(title: str) -> pd.DataFrame:
    """월별 조회수 수집 (Wikimedia REST API)."""
    base = ("https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
            "en.wikipedia.org/all-access/user/{slug}/monthly/20150701/20251231")
    # 일부 문서는 pageviews API에서 _ 대신 - 를 사용
    slugs = [_encode_title(title)]
    alt = _encode_title(title.replace(" ", "-"))
    if alt != slugs[0]:
        slugs.append(alt)
    with _create_session() as session:
        for slug in slugs:
            url = base.format(slug=slug)
            try:
                resp = _request_with_retry(session, url, label=f"pageviews({title})")
                if resp is not None and resp.status_code == 200:
                    data = resp.json()
                    items = data.get("items", [])
                    if items:
                        df = pd.DataFrame(items)
                        # API 슬러그(하이픈 등)가 원래 제목과 다를 수 있으므로 통일
                        if "article" in df.columns:
                            df["article"] = title
                        return df
            except Exception as ex:
                log_error(title, title, f"pageviews 수집 실패: {ex}")
    return pd.DataFrame()


@_with_timeout(90)
def article_info(title: str) -> pd.DataFrame:
    """문서 메타정보 수집 (xtools API)."""
    title_encoded = _encode_title(title)
    url = f"https://xtools.wmcloud.org/api/page/articleinfo/en.wikipedia.org/{title_encoded}?format=json"
    with _create_session() as session:
        try:
            resp = _request_with_retry(session, url, label=f"article_info({title})", timeout=60)
            if resp is not None and resp.status_code == 200:
                data = resp.json()
                if "error" in data:
                    log_error(title, title, f"article_info API 에러: {data['error']}")
                    return pd.DataFrame()
                df = pd.DataFrame.from_dict(data=data, orient="index").T
                # 불필요 컬럼 안전 삭제
                for col in ("elapsed_time",):
                    if col in df.columns:
                        df = df.drop(columns=[col])
                # assessment 안전 처리
                if "assessment" in df.columns:
                    df["assessment"] = df["assessment"].astype(str).str.replace("'", "\\'", regex=False)
                return df
        except Exception as ex:
            log_error(title, title, f"article_info 수집 실패: {ex}")
    return pd.DataFrame()


@_with_timeout(60)
def crawl_link(title: str) -> pd.DataFrame:
    """문서 링크 정보 수집 (xtools API)."""
    title_encoded = _encode_title(title)
    url = f"https://xtools.wmcloud.org/api/page/links/en.wikipedia.org/{title_encoded}"
    with _create_session() as session:
        try:
            resp = _request_with_retry(session, url, label=f"crawl_link({title})")
            if resp is not None and resp.status_code == 200:
                data = resp.json()
                if "error" in data:
                    log_error(title, title, f"crawl_link API 에러: {data['error']}")
                    return pd.DataFrame()
                df = pd.DataFrame.from_dict(data=data, orient="index").T
                for col in ("elapsed_time",):
                    if col in df.columns:
                        df = df.drop(columns=[col])
                return df
        except Exception as ex:
            log_error(title, title, f"crawl_link 수집 실패: {ex}")
    return pd.DataFrame()


# ============================================================
# 3. wiki_info_crawl — 오케스트레이터
# ============================================================

def _safe_filename(name: str) -> str:
    """파일명에 사용할 수 없는 문자를 치환."""
    return re.sub(r'[\\/:*?"<>|]', "_", str(name))


def wiki_info_crawl(split_input, path, on_node=None):
    """
    노드 목록에 대해 4종 데이터(edit, pageviews, info, link)를 수집.
    각 수집 단계를 개별 try-except로 감싸서 부분 성공을 허용.
    """
    os.makedirs(path, exist_ok=True)
    print(path)

    for idx, item in enumerate(split_input):
        safe_item = _safe_filename(item)
        # 이미 4개 파일이 모두 존재하면 건너뛰기
        suffixes = ("edit", "pageviews", "info", "link")
        expected = [os.path.join(path, f"{safe_item}_{s}.xlsx") for s in suffixes]
        if all(os.path.isfile(f) and os.path.getsize(f) > 0 for f in expected):
            print(f" [스킵] {item} ({idx+1}/{len(split_input)}) - 이미 수집됨", file=sys.stderr)
            if on_node:
                on_node(idx=idx, total=len(split_input), node=item,
                        edit=True, pageviews=True, info=True, link=True, skipped=True)
            continue

        # rate limit 방지
        time.sleep(2 if idx > 0 else 1)
        print(f" 단어 수집 중....: {item} ({idx+1}/{len(split_input)})", file=sys.stderr)

        collected = {}

        # --- edit ---
        try:
            edit_df = crawl_edit_by_year(item)
            if not edit_df.empty:
                collected["edit"] = edit_df
        except Exception as ex:
            log_error(item, item, f"edit 수집 실패: {ex}")

        time.sleep(2)

        # --- pageviews ---
        try:
            pv_df = crawl_pageviews(item)
            if not pv_df.empty:
                collected["pageviews"] = pv_df
        except Exception as ex:
            log_error(item, item, f"pageviews 수집 실패: {ex}")

        time.sleep(2)

        # --- info ---
        try:
            info_df = article_info(item)
            if not info_df.empty:
                collected["info"] = info_df
        except Exception as ex:
            log_error(item, item, f"info 수집 실패: {ex}")

        time.sleep(2)

        # --- link ---
        try:
            link_df = crawl_link(item)
            if not link_df.empty:
                collected["link"] = link_df
        except Exception as ex:
            log_error(item, item, f"link 수집 실패: {ex}")

        # 성공한 것만 저장
        for suffix, df in collected.items():
            out_path = os.path.join(path, f"{safe_item}_{suffix}.xlsx")
            df.to_excel(out_path, index=False)

        if not collected:
            log_error(item, item, "4개 수집 모두 실패")

        # 콜백: 매 노드 처리 후 호출
        if on_node:
            on_node(
                idx=idx, total=len(split_input), node=item,
                edit="edit" in collected, pageviews="pageviews" in collected,
                info="info" in collected, link="link" in collected, skipped=False,
            )

    return None


# ============================================================
# 4. 유틸리티 함수 (기존 유지)
# ============================================================

def filter_items(file_path, t_rule, c_rule):
    """필터링 네트워크에서 규칙에 맞는 항목 제거."""
    df = pd.read_excel(file_path)

    if df.index.name and str(df.index.name).strip().lower() in {"from_title", "from title", "from"}:
        df = df.reset_index()

    orig_cols = list(df.columns)
    df.columns = [str(c).strip() for c in df.columns]
    lower_to_orig = {c.lower(): c for c in df.columns}

    def pick(*cands):
        for cand in cands:
            if cand in lower_to_orig:
                return lower_to_orig[cand]
        return None

    col_from = pick("from_title", "from title", "from")
    col_to = pick("to_seealso", "to seealso", "to see also", "to")
    col_cat = pick("category", "categories")

    missing = [name for name, col in {
        "From_title": col_from, "To_seealso": col_to, "Category": col_cat
    }.items() if col is None]
    if missing:
        raise KeyError(
            f"엑셀에 필요한 열이 없습니다: {missing}. "
            f"현재 열들: {orig_cols} -> 정규화 후: {list(df.columns)}"
        )

    if isinstance(t_rule, str):
        t_rule = re.compile(t_rule, re.I)
    if isinstance(c_rule, str):
        c_rule = re.compile(c_rule, re.I)

    t_series = df[col_to].fillna("").astype(str)
    c_series = df[col_cat].fillna("").astype(str)

    t_filter_list = [t for t in t_series if t_rule.search(t) is not None]
    c_filter_list = [c for c in c_series if c_rule.search(c) is not None]

    c_filter_items = df[df[col_cat].isin(c_filter_list)]
    c_filter_to_see_also = list(c_filter_items[col_to].fillna("").astype(str))

    all_filter_list = list(set(c_filter_to_see_also + t_filter_list))

    result = df[~df[col_from].astype(str).isin(all_filter_list)]
    result = result[~result[col_to].astype(str).isin(all_filter_list)]
    result = result.reset_index(drop=True)
    return result


def process_see_also(see_also):
    """See also 위키텍스트에서 링크 목록 추출."""
    idx = see_also.find("*")
    if idx >= 0:
        see_also = see_also[idx:]
    see_also = see_also.replace("[[", '"').replace("]]", '"').replace("{{", '"').replace("}}", '"')
    see_also = re.findall(r'"([^"]*)"', see_also)

    filtered = []
    for sa in see_also:
        if any(kw in sa for kw in ["Category:", "div col end", "Div col end"]):
            continue
        filtered.append(sa.split("|"))
    return filtered


def token_processing(text):
    """텍스트 토크나이징."""
    token_list = []
    for t in text:
        try:
            token_list.append(word_tokenize(re.sub(r"[^a-zA-Z0-9 ]", "", str(t)).lower()))
        except Exception:
            token_list.append(["None"])
    return token_list


# ============================================================
# 5. check_seed — 시드 검증
# ============================================================

def check_seed(dataset):
    """시드 제목들의 실제 위키백과 제목/URL/카테고리/본문을 수집."""
    base_url = "https://en.wikipedia.org"
    wiki_api = "https://en.wikipedia.org/w/api.php"
    row_list = []

    with _create_session() as session:
        for index, title in enumerate(dataset):
            if index > 0:
                time.sleep(2)

            try:
                # 1) 실제 제목 수집
                resp = _request_with_retry(session, f"{base_url}/wiki/{_encode_title(title)}",
                                           label=f"check_seed page({title})")
                if resp is None or resp.status_code != 200:
                    log_error(title, title, f"페이지 접근 실패 (status={getattr(resp, 'status_code', 'None')})")
                    continue

                soup = BeautifulSoup(resp.text, "lxml")
                h1 = soup.select_one('h1[id="firstHeading"]')
                if not h1:
                    log_error(title, title, "h1#firstHeading 없음")
                    continue

                true_title = h1.get_text().strip()
                print(f"================ {true_title} =============================")

                # 2) 실제 URL 수집
                time.sleep(1)
                url_resp = _request_with_retry(
                    session, f"{base_url}/wiki/{_encode_title(true_title)}",
                    label=f"check_seed url({true_title})")
                original_url = url_resp.url if url_resp else ""

                # 3) 카테고리 + 본문 수집
                item_category = ""
                item_content = ""

                time.sleep(1)
                params = {
                    "action": "query",
                    "format": "json",
                    "prop": "categories|extracts",
                    "clshow": "!hidden",
                    "titles": true_title,
                }
                try:
                    api_resp = _request_with_retry(session, wiki_api, label=f"check_seed API({true_title})",
                                                   params=params)
                    api_data = api_resp.json()
                    pages = api_data.get("query", {}).get("pages", {})
                    page_data = list(pages.values())[0] if pages else {}
                except Exception as ex:
                    page_data = {}
                    log_error(true_title, true_title, f"리퀘스트 접근X: {ex}")

                # 카테고리 추출
                try:
                    cats = page_data.get("categories", [])
                    if cats:
                        item_category = "| ".join(
                            ct.get("title", "").replace("Category:", "") for ct in cats
                        )
                except Exception:
                    log_error(title, true_title, "카테고리 접근X")

                # 본문 수집
                try:
                    item_content = page_data.get("extract", "")
                    if item_content:
                        content_soup = BeautifulSoup(item_content, "lxml")
                        # See also / References / External links 이후 내용 제거
                        for section_id in ("See_also", "References", "External_links"):
                            cut_tag = content_soup.select_one(f'span[id="{section_id}"]')
                            if cut_tag:
                                cut_str = str(cut_tag.parent)
                                pos = item_content.find(cut_str)
                                if pos >= 0:
                                    item_content = item_content[:pos]
                                break

                        content_soup = BeautifulSoup(item_content, "lxml")
                        for math_el in content_soup.select("math"):
                            math_el.decompose()
                        item_content = content_soup.get_text().strip()
                        item_content = re.sub(r"[\n\xa0]", " ", item_content)
                except Exception:
                    log_error(title, true_title, "본문 수집 X")

                out_row = [index, title, true_title, item_category, item_content, original_url]
                print(out_row)
                row_list.append(out_row)

            except Exception as ex:
                print(f"Error processing {title}: {ex}")
                log_error(title, title, f"check_seed 전체 에러: {ex}")

    return row_list


# ============================================================
# 6. wiki_crawl — See Also 크롤러 (배치 API + 로컬 파싱)
# ============================================================

def _chunked(lst, size):
    """리스트를 size 크기의 청크로 분할."""
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


def _parse_see_also(wikitext):
    """wikitext에서 See Also 섹션의 [[링크]] 목록 추출."""
    match = re.search(
        r'==\s*See also\s*==(.*?)(?=\n==[^=]|\Z)',
        wikitext, re.DOTALL | re.IGNORECASE,
    )
    if not match:
        return []
    section = match.group(1)
    raw_links = re.findall(r'\[\[([^\]]+)\]\]', section)
    results = []
    for link in raw_links:
        target = link.split("|")[0].strip()   # [[Page|Display]] → Page
        target = target.split("#")[0].strip()  # [[Page#Section]] → Page
        if not target or ":" in target:        # Category:, File: 등 네임스페이스 제외
            continue
        results.append(target)
    return results


def wiki_crawl(seed_list):
    """시드 목록에서 See Also 링크를 배치 추출 (MediaWiki Action API)."""
    row_list = []
    wiki_api = "https://en.wikipedia.org/w/api.php"

    with _create_session() as session:
        # --- Step 1: frontier 노드들의 wikitext 배치 수집 ---
        page_contents = {}   # {actual_title: wikitext}
        for batch in _chunked(seed_list, 50):
            params = {
                "action": "query",
                "titles": "|".join(batch),
                "prop": "revisions",
                "rvprop": "content",
                "rvslots": "main",
                "redirects": "1",
                "format": "json",
                "formatversion": "2",
            }
            resp = _request_with_retry(session, wiki_api, label="wikitext_batch", params=params)
            if resp and resp.status_code == 200:
                data = resp.json()
                for page in data.get("query", {}).get("pages", []):
                    if "missing" in page:
                        continue
                    title = page.get("title", "")
                    revs = page.get("revisions", [])
                    if revs:
                        content = revs[0].get("slots", {}).get("main", {}).get("content", "")
                        page_contents[title] = content

        # --- Step 2: 로컬 파싱으로 See Also 추출 ---
        see_also_map = {}  # {from_title: [target1, target2, ...]}
        for title, content in page_contents.items():
            links = _parse_see_also(content)
            if links:
                see_also_map[title] = links

        # 전체 타겟 목록
        all_targets = list(set(
            link for links in see_also_map.values() for link in links
        ))

        # --- Step 3: 타겟 카테고리 배치 수집 + 리다이렉트 해석 ---
        target_info = {}    # {actual_title: category_str}
        redirect_map = {}   # {original: actual}
        for batch in _chunked(all_targets, 50):
            params = {
                "action": "query",
                "titles": "|".join(batch),
                "prop": "categories",
                "redirects": "1",
                "clshow": "!hidden",
                "cllimit": "500",
                "format": "json",
                "formatversion": "2",
            }
            resp = _request_with_retry(session, wiki_api, label="category_batch", params=params)
            if resp and resp.status_code == 200:
                data = resp.json()
                # 리다이렉트 매핑
                for r in data.get("query", {}).get("redirects", []):
                    redirect_map[r["from"]] = r["to"]
                # 카테고리 수집
                for page in data.get("query", {}).get("pages", []):
                    if "missing" in page:
                        continue
                    t = page.get("title", "")
                    cats = page.get("categories", [])
                    cat_str = "| ".join(c.get("title", "").replace("Category:", "") for c in cats)
                    target_info[t] = cat_str

        # --- Step 4: 결과 조립 ---
        for from_title, targets in see_also_map.items():
            for sa_link in targets:
                actual_title = redirect_map.get(sa_link, sa_link)
                category = target_info.get(actual_title, "")
                row_list.append([from_title, sa_link, actual_title, category, ""])

    return row_list


# ============================================================
# 7. n_char_crawler — BFS 확장 (os.chdir 제거, 절대경로 사용)
# ============================================================

def n_char_crawler(split_list, n, path, on_iter=None):
    """BFS 방식으로 See Also 네트워크를 n차시까지 확장."""
    cols = ["From_title", "Fake_seealso", "To_seealso", "Category", "Contents"]
    all_results = []

    os.makedirs(path, exist_ok=True)

    if not split_list:
        return pd.DataFrame(columns=cols)

    for idx, item in enumerate(split_list):
        df_all = pd.DataFrame(columns=cols)

        # 폴더명 안전화
        safe_item = re.sub(r'[\\/:*?"<>|]', "_", str(item))
        item_dir = os.path.join(path, safe_item)
        os.makedirs(item_dir, exist_ok=True)

        print(idx, [item])

        frontier = [str(item)]
        visited = set()
        last_i = 0
        start_i = 1  # 시작 차시 (기본: 1차시부터)

        # --- 이전 차시 결과 복원: 이미 완료된 차시가 있으면 이어서 진행 ---
        for prev_i in range(1, n + 1):
            prev_file = os.path.join(item_dir, f"{prev_i}_char_crawling.xlsx")
            if os.path.isfile(prev_file) and os.path.getsize(prev_file) > 0:
                try:
                    prev_df = pd.read_excel(prev_file)
                    if prev_df.empty:
                        break
                    # 이전 차시 데이터 누적
                    df_all = (
                        pd.concat([df_all, prev_df], ignore_index=True)
                        .drop_duplicates()
                        .reset_index(drop=True)
                    )
                    # visited 복원: From_title(=해당 차시 frontier) + 이전 visited
                    from_nodes = prev_df["From_title"].dropna().astype(str).tolist()
                    visited.update(from_nodes)
                    # frontier 복원: To_seealso 중 미방문 노드
                    to_list = prev_df["To_seealso"].dropna().astype(str).tolist()
                    frontier = [x for x in dict.fromkeys(to_list) if x and x not in visited]
                    last_i = prev_i
                    start_i = prev_i + 1
                    print(f"  [재개] {prev_i}차시 결과 로드 완료 "
                          f"(수집 {len(prev_df)}건, visited {len(visited)}, "
                          f"다음 frontier {len(frontier)})")
                    if on_iter:
                        on_iter(
                            iteration=prev_i,
                            frontier=len(from_nodes),
                            collected=len(prev_df),
                            visited=len(visited),
                            next_frontier=len(frontier),
                        )
                except Exception as e:
                    print(f"  [재개] {prev_i}차시 파일 로드 실패: {e}, 처음부터 시작")
                    df_all = pd.DataFrame(columns=cols)
                    frontier = [str(item)]
                    visited = set()
                    start_i = 1
                    break
            else:
                break

        if start_i > n:
            print(f"  [재개] {n}차시까지 모두 완료됨, 크롤링 생략")
        elif start_i > 1:
            print(f"  [재개] {start_i}차시부터 이어서 진행 (n={n})")

        for i in range(start_i, n + 1):
            last_i = i
            print(f"{i}차시 시작 (frontier={len(frontier)})")

            if not frontier:
                print(f"더 이상 확장할 seed가 없어 {i-1}차시에서 종료.")
                break

            row_list = wiki_crawl(frontier)
            node_df = pd.DataFrame(row_list, columns=cols)

            visited.update(frontier)

            to_list = node_df["To_seealso"].dropna().astype(str).tolist()
            next_frontier = [x for x in dict.fromkeys(to_list) if x and x not in visited]

            # 중간 저장
            node_df.to_excel(os.path.join(item_dir, f"{i}_char_crawling.xlsx"), index=False)
            print(f"{i}차시입니다. 수집 {len(node_df)}건, 다음 frontier {len(next_frontier)}건")

            df_all = (
                pd.concat([df_all, node_df], ignore_index=True)
                .drop_duplicates()
                .reset_index(drop=True)
            )

            # 콜백: 매 차시 완료 후 호출
            if on_iter:
                on_iter(
                    iteration=i,
                    frontier=len(frontier),
                    collected=len(node_df),
                    visited=len(visited),
                    next_frontier=len(next_frontier),
                )

            frontier = next_frontier

        # 최종 저장
        if last_i > 0:
            df_all.to_excel(os.path.join(item_dir, f"{last_i}차시 확장 최종_결과.xlsx"), index=False)
        all_results.append(df_all)

    if all_results:
        return (
            pd.concat(all_results, ignore_index=True)
            .drop_duplicates()
            .reset_index(drop=True)
        )
    return pd.DataFrame(columns=cols)
