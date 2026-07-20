# pipeline.py
# -*- coding: utf-8 -*-

from __future__ import annotations
import os
import re
import hashlib
import json
import datetime
from pathlib import Path
from typing import List, Dict, Optional
from functools import reduce

import pandas as pd
import numpy as np
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
from sklearn.preprocessing import MinMaxScaler

# ====== 외부 모듈 (같은 폴더에 있어야 함) ======
import wiki_crawling

# ==========
# 경로/유틸
# ==========
BASE_RUN_DIR = "./runs"


def ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)


def exists_nonempty(p: str | Path, min_bytes: int = 1024) -> bool:
    try:
        pp = Path(p)
        return pp.is_file() and pp.stat().st_size >= min_bytes
    except Exception:
        return False


def slugify_seed(s: str) -> str:
    return re.sub(r"\s+", "_", s.strip().replace("/", "-"))


def title_space(s: str) -> str:
    t = re.sub(r"\s+", " ", str(s).replace("_", " ").strip())
    return t[:1].upper() + t[1:] if t else t


def listdir_nods(path: str | Path) -> List[str]:
    p = Path(path)
    if not p.exists() or not p.is_dir():
        return []
    return [f for f in os.listdir(p) if not f.startswith(".")]


def multi_seed_hash(seeds: List[str], n_depth: int) -> str:
    """시드 목록 + depth로 결정적 8자리 해시 생성 (캐시 디렉토리명용)"""
    key = json.dumps(
        {"seeds": sorted(s.strip().lower() for s in seeds), "n": n_depth},
        sort_keys=True,
    )
    return hashlib.sha256(key.encode()).hexdigest()[:8]


# ======================
# 1) True title 확인/저장
# ======================
def resolve_true_title(seed: str, out_xlsx: str) -> pd.DataFrame:
    """@st.cache_data(show_spinner=False) 제거됨"""
    if exists_nonempty(out_xlsx):
        df = pd.read_excel(out_xlsx)
    else:
        print(f"[{seed}] True Title 확인 (API 호출)...")
        rows = wiki_crawling.check_seed([seed])
        df = pd.DataFrame(rows, columns=["No", "Title", "True_title", "Category", "Contents", "URL"])
        df.to_excel(out_xlsx, index=False)
    df["Title"] = df["Title"].map(title_space)
    df["True_title"] = df["True_title"].map(title_space)
    return df


# ============================
# 2) 확장/필터
# ============================

def expand_network(seed_true_title: str, out_xlsx: str, n: int, on_iter=None) -> str:
    """
    확장 결과 파일의 '실제 경로'를 문자열로 반환.
    조기 종료 시(frontier 소진) 실제 완료된 차시 파일을 자동 탐색.
    """
    if exists_nonempty(out_xlsx):
        return out_xlsx

    out_dir = str(Path(out_xlsx).parent)

    # 크롤링 전에 alt 경로 / 조기 종료 파일이 이미 있는지 확인
    alt = Path(out_dir) / seed_true_title / f"{n}차시 확장 최종_결과.xlsx"
    if exists_nonempty(alt):
        print(f"[{seed_true_title}] 기존 확장 파일 발견 (alt): {alt}")
        return str(alt)
    for actual_n in range(n, 0, -1):
        for search_dir in [Path(out_dir), Path(out_dir) / seed_true_title]:
            fallback = search_dir / f"{actual_n}차시 확장 최종_결과.xlsx"
            if exists_nonempty(fallback):
                if actual_n < n:
                    print(f"[{seed_true_title}] 기존 {actual_n}차시 결과 발견: {fallback}")
                return str(fallback)

    ensure_dir(out_dir)

    print(f"[{seed_true_title}] 위키 네트워크 확장 중 (n={n})...")
    wiki_crawling.n_char_crawler([seed_true_title], n, out_dir, on_iter=on_iter)

    # 1) 기본 경로 (n차시)
    if exists_nonempty(out_xlsx):
        return out_xlsx

    # 2) 대체 경로: runs/<seed>/seed_item/<True_title>/<n>차시 확장 최종_결과.xlsx
    alt = Path(out_dir) / seed_true_title / f"{n}차시 확장 최종_결과.xlsx"
    if exists_nonempty(alt):
        return str(alt)

    # 3) 조기 종료: n보다 작은 차시 파일을 역순으로 탐색
    for actual_n in range(n - 1, 0, -1):
        fallback = Path(out_dir) / f"{actual_n}차시 확장 최종_결과.xlsx"
        if exists_nonempty(fallback):
            print(f"[{seed_true_title}] {n}차시 요청했으나 {actual_n}차시에서 조기 종료. {actual_n}차시 결과 사용.")
            return str(fallback)
        fallback_alt = Path(out_dir) / seed_true_title / f"{actual_n}차시 확장 최종_결과.xlsx"
        if exists_nonempty(fallback_alt):
            print(f"[{seed_true_title}] {n}차시 요청했으나 {actual_n}차시에서 조기 종료. {actual_n}차시 결과 사용.")
            return str(fallback_alt)

    # 4) 그래도 없으면 에러
    raise FileNotFoundError(
        f"확장 결과 파일을 찾을 수 없습니다:\n - {out_xlsx}\n - {alt}"
    )


def _read_excel(path: str) -> pd.DataFrame:
    """@st.cache_data(show_spinner=False) 제거됨"""
    return pd.read_excel(path) if exists_nonempty(path, 1) else pd.DataFrame()


def filter_network(
        src_xlsx: str,
        seed_true_title: str,
        out_xlsx: str,
        rule_file: str = "./data/wiki_rule.xlsx",
        mode: str = "balanced",
) -> str:
    """
    wiki_rule.xlsx의 title/category 룰에 따라 네트워크를 필터링한다.
    (st.warning, st.expander 제거됨)
    """

    if exists_nonempty(out_xlsx):
        return out_xlsx

    df = pd.read_excel(src_xlsx)

    # 인덱스 복구
    if df.index.name and str(df.index.name).strip().lower() in {"from_title", "from title", "from"}:
        df = df.reset_index()

    # 열 이름 정규화
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
        raise KeyError(f"엑셀에 필요한 열이 없습니다: {missing}. 현재 열들: {orig_cols}")

    # 값 정규화
    df[col_from] = df[col_from].astype(str).map(title_space)
    df[col_to] = df[col_to].astype(str).map(title_space)
    if col_cat:
        df[col_cat] = df[col_cat].fillna("").astype(str)

    # 룰 로드
    title_rule, cate_rule = [], []
    if mode != "off" and os.path.exists(rule_file):
        try:
            wiki_rule = pd.read_excel(rule_file)
            if {"col", "item"} <= set(wiki_rule.columns):
                title_rule = (
                    wiki_rule[wiki_rule["col"].str.lower() == "title"]["item"].dropna().astype(str).tolist()
                )
                cate_rule = (
                    wiki_rule[wiki_rule["col"].str.lower() == "category"]["item"].dropna().astype(str).tolist()
                )
        except Exception as e:
            print(f"경고: 룰 파일 로드 오류: {e} → 룰 생략")
    else:
        if mode != "off":
            print(f"경고: 룰 파일을 찾을 수 없어({rule_file}) 룰 생략")

    # 정규식 준비
    title_pat = None
    if title_rule:
        pats = [re.escape(title_space(t).lower()) for t in title_rule if str(t).strip()]
        if pats:
            title_pat = re.compile(r"\b(?:%s)\b" % "|".join(pats), re.I)

    cate_set = {c.strip().lower() for c in cate_rule if str(c).strip()}

    # 제외 후보 계산
    t_series = df[col_to].fillna("").astype(str)
    c_series = df[col_cat].fillna("").astype(str) if col_cat else pd.Series([""] * len(df))

    t_filter_list = [t for t in t_series if title_pat and title_pat.search(t.lower())]

    def cat_tokens(s: str):
        return [tok.strip().lower() for tok in s.split("|") if tok.strip()]

    mask_cate = c_series.map(lambda s: any(tok in cate_set for tok in cat_tokens(s)))
    c_filter_to_see_also = list(df.loc[mask_cate, col_to].fillna("").astype(str))

    exclude_titles = set(map(str.lower, t_filter_list + c_filter_to_see_also))

    # seed_true_title 화이트리스트
    seed_norm = title_space(seed_true_title).lower()
    if seed_norm in exclude_titles:
        exclude_titles.remove(seed_norm)

    # 최종 필터링
    norm_from = df[col_from].str.lower()
    norm_to = df[col_to].str.lower()
    keep_mask = ~norm_from.isin(exclude_titles) & ~norm_to.isin(exclude_titles)

    drop_ratio = 1 - float(keep_mask.mean())
    if drop_ratio > 0.9:  # 세이프가드
        print(f"경고: 필터로 {drop_ratio:.1%} 제거 감지 → 룰 스킵")
        keep_mask[:] = True

    result = df[keep_mask].copy()

    # 자기루프 제거 + 중복 통합
    result = result[result[col_from] != result[col_to]]
    result = result.groupby([col_from, col_to]).size().reset_index(name="weight")

    # 저장
    result.rename(columns={col_from: "From_title", col_to: "To_seealso"}, inplace=True)
    result.to_excel(out_xlsx, index=False)

    print(f"필터링 요약: 원본: {len(df):,} → 필터 후: {len(result):,} ")
    return out_xlsx


# =========================================
# 3) Xtools 수집 + 통합
# =========================================

def xtools_collect(seed_true_title: str, filtered_xlsx: str, xtools_dir: str, flag_file: str,
                   on_node=None, on_detail=None):
    """수집은 파일 여러 개를 만들므로 센티넬(.collect_done)로 스킵. (st.spinner 제거)"""
    if exists_nonempty(flag_file, min_bytes=1):
        return
    ensure_dir(xtools_dir)
    df = pd.read_excel(filtered_xlsx) if exists_nonempty(filtered_xlsx, 1) else pd.DataFrame()
    nodes = set(map(title_space, list(df.get("From_title", [])) + list(df.get("To_seealso", []))))
    # 시드 자체를 항상 포함 (See Also가 없어서 엣지가 비어 있어도 수집 대상)
    nodes.add(title_space(seed_true_title))
    nodes = [n for n in nodes if str(n).strip()]
    if on_detail:
        on_detail("collect_start", {"nodes": nodes, "total": len(nodes)})
    if nodes:
        print(f"[{seed_true_title}] 편집/정보/링크/조회수 수집 중 ({len(nodes)}개 노드)...")
        wiki_crawling.wiki_info_crawl(nodes, xtools_dir, on_node=on_node)

        # --- 누락 노드 재시도 (1회) ---
        suffixes = ("edit", "pageviews", "info", "link")
        incomplete = [
            n for n in nodes
            if not all(
                os.path.isfile(os.path.join(xtools_dir, f"{wiki_crawling._safe_filename(n)}_{s}.xlsx"))
                and os.path.getsize(os.path.join(xtools_dir, f"{wiki_crawling._safe_filename(n)}_{s}.xlsx")) > 0
                for s in suffixes
            )
        ]
        if incomplete:
            print(f"[{seed_true_title}] 누락 노드 {len(incomplete)}개 재시도...")
            wiki_crawling.wiki_info_crawl(incomplete, xtools_dir, on_node=None)

    if on_detail:
        on_detail("collect_done", {"total": len(nodes)})
    Path(flag_file).write_text("done", encoding="utf-8")


def xtools_integrate(seed_true_title: str, xtools_dir: str) -> Dict[str, str]:
    """
    편집/조회/링크/정보 '원본(row)'과 '통합(wide)' 포맷 혼재를 흡수.
    _all_*.xlsx 통합본은 입력으로 스킵.
    개별 파일이 통합본보다 새로우면 재생성.
    """
    files = listdir_nods(xtools_dir)
    edit_pat = re.compile(r"(?<!_all)_edit\.xlsx$")
    info_pat = re.compile(r"(?<!_all)_info\.xlsx$")
    link_pat = re.compile(r"(?<!_all)_link\.xlsx$")
    view_pat = re.compile(r"(?<!_all)_pageviews\.xlsx$")

    def _needs_rebuild(out_path: str, pat: re.Pattern) -> bool:
        """출력 파일이 없거나 개별 파일 중 하나라도 더 새로우면 True"""
        if not exists_nonempty(out_path):
            return True
        out_mtime = os.path.getmtime(out_path)
        for f in files:
            if pat.search(f):
                fp = os.path.join(xtools_dir, f)
                try:
                    if os.path.getmtime(fp) > out_mtime:
                        return True
                except Exception:
                    pass
        return False

    def _ensure_title(df: pd.DataFrame) -> pd.DataFrame:
        if "title" not in df.columns:
            if df.index.name and str(df.index.name).lower() == "title":
                df = df.reset_index()
        if "title" not in df.columns and "page" in df.columns:
            df = df.rename(columns={"page": "title"})
        if "title" not in df.columns and "article" in df.columns:
            df = df.rename(columns={"article": "title"})
        if "title" not in df.columns:
            candidates = [c for c in df.columns if "title" in c.lower() or "page" in c.lower()]
            if candidates:
                df = df.rename(columns={candidates[0]: "title"})
        if "title" not in df.columns:
            raise KeyError(f"'title' 컬럼 없음. cols={list(df.columns)}")
        df["title"] = df["title"].map(title_space)
        return df

    # ---- edit
    edit_frames = []
    for f in files:
        if not edit_pat.search(f):
            continue
        df = pd.read_excel(os.path.join(xtools_dir, f))
        df = _ensure_title(df)
        if ("edits" in df.columns) and (("year" in df.columns) or ("timestamp" in df.columns)):
            if "year" not in df.columns and "timestamp" in df.columns:
                df["year"] = df["timestamp"].astype(str).str[:4]
            df["year"] = df["year"].astype(str)
            piv = df.pivot(index="title", columns="year", values="edits")
            edit_frames.append(piv)
            continue
        year_cols = [str(c) for c in df.columns if re.fullmatch(r"\d{4}", str(c))]
        year_cols = [c for c in year_cols if 2000 <= int(c) <= 2100]
        if year_cols:
            edit_frames.append(df[["title"] + year_cols].set_index("title"))
    edit_out = os.path.join(xtools_dir, f"{seed_true_title}_all_edit.xlsx")
    if _needs_rebuild(edit_out, edit_pat):
        if edit_frames:
            e = pd.concat(edit_frames, axis=0, ignore_index=False)
            e = e.groupby(e.index).max()
            e = e.reset_index()
        else:
            e = pd.DataFrame(columns=["title"])
        e.sort_values("title").to_excel(edit_out, index=False)

    # ---- info
    info_out = os.path.join(xtools_dir, f"{seed_true_title}_all_info.xlsx")
    if _needs_rebuild(info_out, info_pat):
        info_df = pd.DataFrame()
        for f in files:
            if not info_pat.search(f):
                continue
            df = pd.read_excel(os.path.join(xtools_dir, f))
            df["page"] = f[:-10].replace("_", " ")
            df = _ensure_title(df)
            info_df = pd.concat([info_df, df], ignore_index=True)
        (info_df.sort_values("title") if not info_df.empty else pd.DataFrame(columns=["title"])) \
            .to_excel(info_out, index=False)

    # ---- link
    link_out = os.path.join(xtools_dir, f"{seed_true_title}_all_link.xlsx")
    if _needs_rebuild(link_out, link_pat):
        link_frames = []
        for f in files:
            if not link_pat.search(f):
                continue
            df = pd.read_excel(os.path.join(xtools_dir, f))
            df = _ensure_title(df)
            link_frames.append(df)
        link_df = pd.concat(link_frames, ignore_index=True) if link_frames else pd.DataFrame(columns=["title"])
        link_df.sort_values("title").to_excel(link_out, index=False)

    # ---- pageviews
    view_out = os.path.join(xtools_dir, f"{seed_true_title}_all_pageviews.xlsx")
    if _needs_rebuild(view_out, view_pat):
        view_frames = []
        for f in files:
            if not view_pat.search(f):
                continue
            df = pd.read_excel(os.path.join(xtools_dir, f))
            if "article" in df.columns and "title" not in df.columns:
                df = df.rename(columns={"article": "title"})
            df = _ensure_title(df)
            if "timestamp" in df.columns and "year" not in df.columns:
                df["year"] = df["timestamp"].astype(str).str[:4]
            if "views" not in df.columns:
                continue
            df["year"] = df["year"].astype(str)
            view_frames.append(df[["title", "year", "views"]])
        if view_frames:
            v = pd.concat(view_frames, ignore_index=True)
            vp = v.pivot_table(index="title", columns="year", values="views", aggfunc="sum", fill_value=0)
            vp = vp.reset_index()
        else:
            vp = pd.DataFrame(columns=["title"])
        vp.sort_values("title").to_excel(view_out, index=False)

    return {"edit": edit_out, "info": info_out, "link": link_out, "pageviews": view_out}


# ===============
# 3-b) 네트워크 시각화 유틸 (API 포맷팅용)
# ===============

def _subgraph_for_viz(edges: pd.DataFrame, pr_df: pd.DataFrame, top_n: int, degree_min: int) -> nx.Graph:
    """PR 상위 노드 + 이웃을 중심으로 부분그래프를 만들고, 최소 차수 필터를 적용합니다."""
    if edges.empty:
        return nx.Graph()
    G = nx.from_pandas_edgelist(edges, "From_title", "To_seealso", edge_attr="weight", create_using=nx.Graph())
    if pr_df is not None and not pr_df.empty and "title" in pr_df.columns:
        base_nodes = [n for n in pr_df["title"].head(int(top_n)).tolist() if n in G]
    else:
        base_nodes = list(G.nodes)[: int(top_n)]
    keep = set()
    for n in base_nodes:
        keep.add(n)
        keep.update(G.neighbors(n))
    if degree_min and degree_min > 0:
        deg = dict(G.degree())
        keep |= {n for n, d in deg.items() if d >= degree_min}
    H = G.subgraph(keep).copy()
    if H.number_of_nodes() < 3:  # 너무 작으면 원그래프 사용
        H = G
    return H


def _compute_communities(H: nx.Graph) -> Dict[str, int]:
    """greedy modularity 기반 커뮤니티 라벨을 계산합니다."""
    try:
        comms = list(greedy_modularity_communities(H))
        label = {}
        for i, c in enumerate(comms):
            for n in c:
                label[n] = i
        if not label:
            label = {n: 0 for n in H.nodes}
        return label
    except Exception:
        return {n: 0 for n in H.nodes}


# ===============
# 4) PageRank
# ===============

def load_edges(xlsx: str) -> pd.DataFrame:
    df = pd.read_excel(xlsx)
    lower = {c.lower(): c for c in df.columns}
    fcol = lower.get("from_title") or lower.get("from title") or lower.get("from")
    tcol = lower.get("to_seealso") or lower.get("to seealso") or lower.get("to see also") or lower.get("to")
    if not fcol or not tcol:
        raise KeyError(f"필수 열(From_title/To_seealso) 없음. cols={df.columns.tolist()}")
    out = df[[fcol, tcol]].rename(columns={fcol: "From_title", tcol: "To_seealso"}).copy()
    out["From_title"] = out["From_title"].map(title_space)
    out["To_seealso"] = out["To_seealso"].map(title_space)
    out = out[out["From_title"] != out["To_seealso"]]
    out = out.groupby(["From_title", "To_seealso"]).size().reset_index(name="weight")
    return out


def compute_pagerank(filtered_xlsx: str, out_xlsx: str,
                     seed_nodes: Optional[List[str]] = None) -> pd.DataFrame:
    """PageRank 계산. 엣지가 없는 시드는 pagerank=1.0으로 포함."""
    if exists_nonempty(out_xlsx):
        return pd.read_excel(out_xlsx)
    edges = load_edges(filtered_xlsx)
    if edges.empty:
        # 엣지가 없어도 시드 노드는 pagerank=1.0으로 포함
        if seed_nodes:
            pr_df = pd.DataFrame({
                "title": [title_space(s) for s in seed_nodes],
                "pagerank": [1.0] * len(seed_nodes),
            })
        else:
            pr_df = pd.DataFrame(columns=["title", "pagerank"])
        pr_df.to_excel(out_xlsx, index=False)
        return pr_df
    G = nx.from_pandas_edgelist(edges, "From_title", "To_seealso", edge_attr="weight", create_using=nx.DiGraph())
    try:
        pr = nx.pagerank(G, alpha=0.85, max_iter=200, weight="weight")
    except nx.PowerIterationFailedConvergence:
        pr = nx.pagerank(G, alpha=0.80, max_iter=400, weight="weight", tol=1e-6)
    pr_df = (
        pd.DataFrame.from_dict(pr, orient="index", columns=["pagerank"]).reset_index().rename(
            columns={"index": "title"})
    )
    # 엣지에 없는 시드 노드를 최소 pagerank로 추가
    if seed_nodes:
        existing = set(pr_df["title"].values)
        missing = [title_space(s) for s in seed_nodes if title_space(s) not in existing]
        if missing:
            min_pr = float(pr_df["pagerank"].min()) if not pr_df.empty else 0.0
            supplement = pd.DataFrame({"title": missing, "pagerank": [min_pr] * len(missing)})
            pr_df = pd.concat([pr_df, supplement], ignore_index=True)
    pr_df = pr_df.sort_values("pagerank", ascending=False).reset_index(drop=True)
    pr_df.to_excel(out_xlsx, index=False)
    return pr_df


# =========================
# 5) 지표 집계 & 정규화
# =========================

def compute_statistics(
        edit_xlsx: str,
        pageviews_xlsx: str,
        link_xlsx: str,
        info_xlsx: str,
        pr_df: pd.DataFrame,
        out_xlsx: str,
        ref_year: Optional[str] = None,
) -> pd.DataFrame:
    # 호출부(_need_recompute)에서 재계산 여부를 결정하므로 내부 스킵 제거

    # --- edit (이하 동일)
    edit_df = _read_excel(edit_xlsx)
    if edit_df.empty:
        edit_df = pd.DataFrame(columns=["title"])  # 안전
    edit_df.columns = edit_df.columns.map(str)
    num_cols = [c for c in edit_df.columns if c != "title"]
    edit_df["총편집수"] = edit_df[num_cols].select_dtypes(include="number").sum(axis=1)
    if ref_year is None:
        year_cols = [c for c in num_cols if str(c).isdigit()]
        ref_year = max(year_cols) if year_cols else None
    if ref_year and ref_year in edit_df.columns:
        denom = edit_df["총편집수"].replace(0, np.nan)
        edit_df["공급부상성"] = edit_df[ref_year] / denom
    else:
        edit_df["공급부상성"] = 0.0

    # --- pageviews (이하 동일)
    pv_df = _read_excel(pageviews_xlsx)
    if pv_df.empty:
        pv_df = pd.DataFrame(columns=["title"])  # 안전
    pv_df.columns = pv_df.columns.map(str)
    pv_num_cols = [c for c in pv_df.columns if c != "title"]
    pv_df["총조회수"] = pv_df[pv_num_cols].select_dtypes(include="number").sum(axis=1)
    if ref_year and ref_year in pv_df.columns:
        denom = pv_df["총조회수"].replace(0, np.nan)
        pv_df["수요부상성"] = pv_df[ref_year] / denom
    else:
        pv_df["수요부상성"] = 0.0

    # --- link (이하 동일)
    link_df = _read_excel(link_xlsx)
    if not link_df.empty and "title" in link_df.columns:
        if "links_in_count" in link_df.columns:
            link_df = link_df[["title", "links_in_count"]].rename(columns={"links_in_count": "확산성"})
        elif "links_in" in link_df.columns:
            link_df = link_df[["title", "links_in"]].rename(columns={"links_in": "확산성"})
        else:
            link_df = link_df[["title"]].copy();
            link_df["확산성"] = 0
    else:
        link_df = pd.DataFrame(columns=["title", "확산성"])  # 안전

    # --- info (이하 동일)
    info_df = _read_excel(info_xlsx)
    if not info_df.empty and "title" in info_df.columns:
        col_map = {}
        if "created_at" in info_df.columns: col_map["created_at"] = "생성일"
        if "modified_at" in info_df.columns: col_map["modified_at"] = "최근편집일"
        if col_map:
            info_df = info_df[["title"] + list(col_map.keys())].rename(columns=col_map)
        else:
            info_df = info_df[["title"]].copy();
            info_df["생성일"] = "";
            info_df["최근편집일"] = ""
    else:
        info_df = pd.DataFrame(columns=["title", "생성일", "최근편집일"])  # 안전

    frames = [
        edit_df[["title", "공급부상성", "총편집수"]] if not edit_df.empty else pd.DataFrame(columns=["title", "공급부상성", "총편집수"]),
        pv_df[["title", "수요부상성", "총조회수"]] if not pv_df.empty else pd.DataFrame(columns=["title", "수요부상성", "총조회수"]),
        link_df[["title", "확산성"]] if not link_df.empty else pd.DataFrame(columns=["title", "확산성"]),
        info_df[["title", "생성일", "최근편집일"]] if not info_df.empty else pd.DataFrame(columns=["title", "생성일", "최근편집일"]),
    ]

    # inner merge 대신 outer→누락 보호
    core = reduce(lambda x, y: pd.merge(x, y, on="title", how="outer"), frames)

    pr_use = pr_df.rename(columns={"pagerank": "기술집약도"}) if not pr_df.empty else pd.DataFrame(
        columns=["title", "기술집약도"])  # type: ignore

    # --- [수정된 부분] ---
    stats = pd.merge(core, pr_use[["title", "기술집약도"]] if not pr_use.empty else pd.DataFrame(columns=["title", "기술집약도"]),
                     on="title", how="outer")

    # 숫자 컬럼은 0으로, 날짜/문자 컬럼은 ""(빈문자)로 채움
    num_cols_to_fill = ["공급부상성", "수요부상성", "확산성", "기술집약도", "총편집수", "총조회수"]
    str_cols_to_fill = ["생성일", "최근편집일"]

    for col in num_cols_to_fill:
        if col in stats.columns:
            stats[col] = stats[col].fillna(0)
    for col in str_cols_to_fill:
        if col in stats.columns:
            stats[col] = stats[col].fillna("")  # 0 대신 ""

    # 컬럼 순서 고정
    all_cols = ["title", "공급부상성", "수요부상성", "확산성", "기술집약도", "총편집수", "총조회수", "생성일", "최근편집일"]
    stats = stats.reindex(columns=[c for c in all_cols if c in stats.columns])
    # --- [여기까지 수정] ---

    stats.to_excel(out_xlsx, index=False)
    return stats

def finalize_merged(
        stats_xlsx: str,
        pagerank_xlsx: str,
        merged_outs: Dict[str, str],
        out_dir: str,
        seed_titles: Optional[List[str]] = None,
) -> Dict[str, str]:
    """통합 후 모든 파일에 공통으로 존재하는 아이템만 추출해 최종 파일 생성.

    교집합 기준: edit ∩ pageviews ∩ info.
    pagerank는 교집합 기준에 포함하지 않고 결과 필터링만 적용.
    """
    ensure_dir(out_dir)

    pr_df = _read_excel(pagerank_xlsx)
    edit_df = _read_excel(merged_outs.get("edit", ""))
    pv_df = _read_excel(merged_outs.get("pageviews", ""))
    info_df = _read_excel(merged_outs.get("info", ""))
    stats_df = _read_excel(stats_xlsx)

    # 교집합: edit ∩ pageviews ∩ info
    sets = []
    labels = []
    for label, df in [("edit", edit_df), ("pageviews", pv_df), ("info", info_df)]:
        if not df.empty and "title" in df.columns:
            sets.append(set(df["title"].dropna().map(title_space)))
            labels.append(label)

    if not sets:
        print("[finalize] 교집합 계산 불가 — 데이터 없음")
        return {}

    common_titles = set.intersection(*sets)
    parts = ", ".join(f"{l}={len(s)}" for l, s in zip(labels, sets))
    print(f"[finalize] 교집합: {len(common_titles)}개 ({parts})")

    if not common_titles:
        print("[finalize] 교집합이 비어 있습니다.")
        return {}

    final_paths = {}

    # 각 파일을 교집합으로 필터
    for label, df, fname in [
        ("stats", stats_df, "final_statistics.xlsx"),
        ("pagerank", pr_df, "final_pagerank.xlsx"),
        ("edit", edit_df, "final_all_edit.xlsx"),
        ("pageviews", pv_df, "final_all_pageviews.xlsx"),
    ]:
        if df.empty or "title" not in df.columns:
            continue
        df = df[df["title"].map(title_space).isin(common_titles)].copy()
        out_path = os.path.join(out_dir, fname)
        df.to_excel(out_path, index=False)
        final_paths[label] = out_path
        print(f"  {fname}: {len(df)}건")

    # info, link — 있으면 필터, 없어도 에러 안 남
    for label in ("info", "link"):
        src = merged_outs.get(label, "")
        df = _read_excel(src)
        if df.empty or "title" not in df.columns:
            continue
        df = df[df["title"].map(title_space).isin(common_titles)].copy()
        out_path = os.path.join(out_dir, f"final_all_{label}.xlsx")
        df.to_excel(out_path, index=False)
        final_paths[label] = out_path
        print(f"  final_all_{label}.xlsx: {len(df)}건")

    # --- 시드 포함 여부 검증 ---
    if seed_titles:
        common_lower = {t.lower() for t in common_titles}
        found = []
        missing = []
        for s in seed_titles:
            s_norm = title_space(s)
            if s_norm in common_titles or s_norm.lower() in common_lower:
                found.append(s_norm)
            else:
                missing.append(s_norm)
        print(f"[finalize] 시드 검증: {len(found)}/{len(seed_titles)}개 포함")
        if missing:
            print(f"[finalize] 누락 시드: {missing}")
        final_paths["_seed_check"] = {
            "total": len(seed_titles),
            "found": len(found),
            "missing": missing,
        }

    return final_paths


def normalize_indicators(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    scaler = MinMaxScaler()
    out[cols] = out[cols].fillna(0)
    if len(out) == 0:
        for c in cols:
            out[c + "_norm"] = 0
        return out
    out[[c + "_norm" for c in cols]] = scaler.fit_transform(out[cols])
    return out


def score_1to5(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """전역 MinMax 후 1~5점 매핑(상수열이면 3점)."""
    out = df.copy()
    for c in cols:
        s = out[c].astype(float).fillna(0.0)
        vmin, vmax = float(s.min()), float(s.max())
        if vmax > vmin:
            out[c + "_score"] = 1.0 + 4.0 * (s - vmin) / (vmax - vmin)
        else:
            out[c + "_score"] = 3.0
    return out


# ===============
# 연도별 추세 데이터 추출 (API 포맷팅용)
# ===============

def extract_year_series(df: pd.DataFrame, title: str) -> pd.Series:
    """[plot_yearly_trends] 함수에서 분리된 헬퍼 함수"""
    if df.empty or "title" not in df.columns:
        return pd.Series(dtype=float)
    row = df.loc[df["title"] == title]
    if row.empty:
        return pd.Series(dtype=float)
    year_cols = [c for c in row.columns if re.fullmatch(r"\d{4}", str(c))]
    if not year_cols:
        return pd.Series(dtype=float)
    year_cols = sorted(year_cols, key=lambda x: int(x))
    s = row[year_cols].iloc[0]
    s = pd.to_numeric(s, errors="coerce").fillna(0.0)
    s.index = pd.Index([int(y) for y in s.index], name="year")
    return s


# ===============================================
# API를 위한 메인 파이프라인 및 포맷팅 함수
# ===============================================

# pipeline.py (약 656번째 줄)

def run_analysis_pipeline(seed: str, n_depth: int, use_existing: bool = True,
                         on_progress=None, on_detail=None):
    """
    app2.py의 'if run:' 블록에서 Streamlit UI 코드를 제거한
    핵심 데이터 생성 파이프라인입니다.

    사용자의 캐싱 요구사항('use_existing=True')을 따릅니다.
    on_progress: 선택적 콜백 함수 (step: int, total: int, message: str)
    on_detail: 선택적 상세 콜백 함수 (event: str, data: dict)
    """

    def _report(step, total, msg):
        print(msg)
        if on_progress:
            on_progress(step, total, msg)

    def _detail(event, data=None):
        if on_detail:
            on_detail(event, data or {})

    if not seed.strip():
        raise ValueError("시드를 입력하세요.")

    print(f"[{seed}] 파이프라인 시작 (n_depth={n_depth})")

    # 세션 상태 대신 로컬 변수 사용
    seed_slug = slugify_seed(seed)
    base_dir = Path(BASE_RUN_DIR) / seed_slug
    seed_item_dir = base_dir / "seed_item"
    xtools_dir = base_dir / "xtools_item" / seed_slug
    ensure_dir(base_dir)
    ensure_dir(seed_item_dir)
    ensure_dir(xtools_dir)

    # 파일 경로들
    paths = {
        "true": str(base_dir / "true_title.xlsx"),
        "expand": str(seed_item_dir / f"{n_depth}차시 확장 최종_결과.xlsx"),
        "filter": str(seed_item_dir / f"{seed_slug}_filtering_network.xlsx"),
        "xtools_dir": str(xtools_dir),
        "collect_flag": str(xtools_dir / ".collect_done"),
        "edit": str(xtools_dir / f"{seed_slug}_all_edit.xlsx"),
        "info": str(xtools_dir / f"{seed_slug}_all_info.xlsx"),
        "link": str(xtools_dir / f"{seed_slug}_all_link.xlsx"),
        "pageviews": str(xtools_dir / f"{seed_slug}_all_pageviews.xlsx"),
        "pagerank": str(base_dir / f"{seed_slug}_pagerank.xlsx"),
        "stats": str(base_dir / f"{seed_slug}_statistics.xlsx"),
    }
    ensure_dir(Path(paths["xtools_dir"]))

    # --- _need_recompute 헬퍼 함수 정의 (함수 내부에 위치) ---
    def _need_recompute(inp: str, out: str) -> bool:
        # 출력 파일이 없으면 재계산 필요
        if not exists_nonempty(out, min_bytes=1):
            return True
        try:
            # 입력(inp) 파일이 출력(out) 파일보다 최신이면 재계산 필요
            return os.path.getmtime(inp) > os.path.getmtime(out)
        except Exception:
            return False

    # --- 헬퍼 함수 정의 끝 ---

    # 1 True title [수정됨]
    _report(1, 7, f"[{seed}] 1/7 시드 확인 중…")
    if not exists_nonempty(paths["true"]) or not use_existing:
        print(f"[{seed}] - 신규 True Title 실행")
        df_true = resolve_true_title(seed, paths["true"])
    else:
        print(f"[{seed}] - 기존 true_title 파일 사용 ({paths['true']})")
        # resolve_true_title은 title_space 정규화를 보장하므로 항상 사용
        df_true = resolve_true_title(seed, paths["true"])

    # 빈 DataFrame 방어: API 실패로 결과가 비어있으면 재시도
    if df_true.empty or "True_title" not in df_true.columns:
        print(f"[{seed}] - true_title이 비어있음, 재시도...")
        # 기존 파일 삭제 후 재수집
        import time
        true_path = Path(paths["true"])
        if true_path.exists():
            true_path.unlink()
        time.sleep(2)  # rate limit 대기
        df_true = resolve_true_title(seed, paths["true"])

    if df_true.empty:
        raise ValueError(f"'{seed}' 시드의 True Title을 확인할 수 없습니다. 위키피디아 API 연결을 확인하세요.")

    seed_true = title_space(str(df_true.loc[0, "True_title"]))
    seed_url = df_true.loc[0, "URL"] if "URL" in df_true.columns else ""
    seed_summary = df_true.loc[0, "Contents"] if "Contents" in df_true.columns else ""

    # 2 확장 [수정됨 - 조기 종료 파일도 탐색]
    _report(2, 7, f"[{seed}] 2/7 네트워크 확장 중 (n={n_depth})...")

    path_expand_primary = paths["expand"]
    expand_dir = Path(path_expand_primary).parent
    path_expand_alt = str(expand_dir / seed_true / Path(path_expand_primary).name)

    # 1. Check alt path (n차시)
    if exists_nonempty(path_expand_alt) and use_existing:
        print(f"[{seed}] - 기존 확장 파일 사용 (alt 경로) ({path_expand_alt})")
        paths["expand"] = path_expand_alt
    # 2. Check primary path (n차시)
    elif exists_nonempty(path_expand_primary) and use_existing:
        print(f"[{seed}] - 기존 확장 파일 사용 (기본 경로) ({path_expand_primary})")
    # 3. Check early-terminated files (n-1, n-2, ... 1차시)
    elif use_existing:
        found = False
        for actual_n in range(n_depth - 1, 0, -1):
            for search_dir in [expand_dir, expand_dir / seed_true]:
                fallback = search_dir / f"{actual_n}차시 확장 최종_결과.xlsx"
                if exists_nonempty(fallback):
                    print(f"[{seed}] - 조기 종료: {actual_n}차시 결과 사용 ({fallback})")
                    paths["expand"] = str(fallback)
                    found = True
                    break
            if found:
                break
        if not found:
            print(f"[{seed}] - 신규 확장 실행")
            def _on_iter_cb(**kwargs):
                _detail("expand_iter", kwargs)
            paths["expand"] = expand_network(seed_true, path_expand_primary, n_depth, on_iter=_on_iter_cb)
    # 4. Run expansion
    else:
        print(f"[{seed}] - 신규 확장 실행")
        def _on_iter_cb(**kwargs):
            _detail("expand_iter", kwargs)
        paths["expand"] = expand_network(seed_true, path_expand_primary, n_depth, on_iter=_on_iter_cb)

    # 확장 완료 요약 이벤트
    try:
        _expand_df = pd.read_excel(paths["expand"]) if exists_nonempty(paths["expand"], 1) else pd.DataFrame()
        _expand_edges = len(_expand_df)
        _expand_nodes = len(set(
            _expand_df.get("From_title", pd.Series()).dropna().tolist()
            + _expand_df.get("To_seealso", pd.Series()).dropna().tolist()
        )) if not _expand_df.empty else 0
        _detail("expand_done", {"edges": _expand_edges, "nodes": _expand_nodes})
    except Exception:
        pass

    # 3 필터 [수정됨 - _need_recompute 사용]
    _report(3, 7, f"[{seed}] 3/7 네트워크 필터링 중...")
    _before_filter = 0
    try:
        _bf = pd.read_excel(paths["expand"]) if exists_nonempty(paths["expand"], 1) else pd.DataFrame()
        _before_filter = len(_bf)
    except Exception:
        pass

    if _need_recompute(paths["expand"], paths["filter"]) or not use_existing:
        print(f"[{seed}] - 신규 필터링 실행")
        paths["filter"] = filter_network(paths["expand"], seed_true, paths["filter"], mode="balanced")
    else:
        print(f"[{seed}] - 기존 필터 파일 사용 ({paths['filter']})")

    try:
        _af = pd.read_excel(paths["filter"]) if exists_nonempty(paths["filter"], 1) else pd.DataFrame()
        _after_filter = len(_af)
        _filter_nodes = len(set(
            _af.get("From_title", pd.Series()).dropna().tolist()
            + _af.get("To_seealso", pd.Series()).dropna().tolist()
        )) if not _af.empty else 0
        _detail("filter_done", {
            "before": _before_filter,
            "after": _after_filter,
            "removed_pct": round((1 - _after_filter / _before_filter) * 100, 1) if _before_filter > 0 else 0,
            "nodes": _filter_nodes,
        })
    except Exception:
        pass

    # 4 수집 [수정됨 - _need_recompute 사용]
    _report(4, 7, f"[{seed}] 4/7 XTools 수집 중 (가장 오래 걸림)...")

    def _on_node_cb(**kwargs):
        _detail("collect_node", kwargs)

    if _need_recompute(paths["filter"], paths["collect_flag"]) or not use_existing:
        print(f"[{seed}] - 신규 수집 실행")
        xtools_collect(seed_true, paths["filter"], paths["xtools_dir"], paths["collect_flag"],
                       on_node=_on_node_cb, on_detail=_detail)
    else:
        print(f"[{seed}] - 기존 수집 플래그 사용 ({paths['collect_flag']})")

    # 5 통합 (수정 불필요 - 항상 실행)
    _report(5, 7, f"[{seed}] 5/7 XTools 통합 중...")
    outs = xtools_integrate(seed_true, paths["xtools_dir"])

    # 6 PageRank [수정됨 - _need_recompute 사용]
    _report(6, 7, f"[{seed}] 6/7 PageRank 계산 중...")
    if _need_recompute(paths["filter"], paths["pagerank"]) or not use_existing:
        print(f"[{seed}] - 신규 PageRank 계산 실행")
        pr_df = compute_pagerank(paths["filter"], paths["pagerank"], seed_nodes=[seed_true])
    else:
        print(f"[{seed}] - 기존 PageRank 파일 사용 ({paths['pagerank']})")
        pr_df = _read_excel(paths["pagerank"])  # 파일이 있어도 읽어야 함

    # PageRank 상위 5개 미리보기
    if not pr_df.empty:
        top5 = pr_df.head(5)[["title", "pagerank"]].to_dict("records")
        _detail("pagerank_done", {"top5": top5, "total": len(pr_df)})

    # 7 통계 [수정됨 - xtools 통합 파일 변경도 감지]
    _report(7, 7, f"[{seed}] 7/7 통계 집계 중...")
    _stats_need = _need_recompute(paths["pagerank"], paths["stats"])
    if not _stats_need and exists_nonempty(paths["stats"], min_bytes=1):
        # xtools 통합 파일(_all_*)이 stats보다 새로우면 재계산
        _stats_mtime = os.path.getmtime(paths["stats"])
        for _xf in (outs.get("edit"), outs.get("pageviews"), outs.get("link"), outs.get("info")):
            try:
                if _xf and os.path.isfile(_xf) and os.path.getmtime(_xf) > _stats_mtime:
                    _stats_need = True
                    break
            except Exception:
                pass
    if _stats_need or not use_existing:
        print(f"[{seed}] - 신규 통계 집계 실행")
        compute_statistics(
            edit_xlsx=outs.get("edit", paths["edit"]),
            pageviews_xlsx=outs.get("pageviews", paths["pageviews"]),
            link_xlsx=outs.get("link", paths["link"]),
            info_xlsx=outs.get("info", paths["info"]),
            pr_df=pr_df,  # 6단계에서 로드/계산한 pr_df 사용
            out_xlsx=paths["stats"],
            ref_year=None,
        )
    else:
        print(f"[{seed}] - 기존 통계 파일 사용 ({paths['stats']})")

    # 최종 통계 요약
    try:
        _stats_df = _read_excel(paths["stats"])
        if not _stats_df.empty:
            _detail("stats_done", {
                "total_items": len(_stats_df),
                "avg_tech": round(float(_stats_df["기술집약도"].mean()), 6) if "기술집약도" in _stats_df.columns else 0,
                "avg_supply": round(float(_stats_df["공급부상성"].mean()), 6) if "공급부상성" in _stats_df.columns else 0,
                "avg_demand": round(float(_stats_df["수요부상성"].mean()), 6) if "수요부상성" in _stats_df.columns else 0,
            })
    except Exception:
        pass

    print(f"[{seed}] 파이프라인 완료.")

    # API 포맷팅에 필요한 모든 결과물 반환
    return paths, outs, seed_true, seed_url, seed_summary

# --- API 출력 포맷팅 함수들 ---

def format_item_1_entry(seed_true, seed_url):
    return {"item": seed_true, "url": seed_url}


def format_item_2_summary(seed_summary):
    preview = (seed_summary or "").strip()
    if len(preview) > 320:
        preview = preview[:320] + "…"
    return {"summary": preview if preview else "설명을 불러오지 못했습니다."}


def format_item_3_network(filtered_xlsx: str, pr_df: pd.DataFrame):
    """네트워크를 JSON (nodes, edges) 형식으로 변환합니다."""
    edges = load_edges(filtered_xlsx)
    if edges.empty:
        return {"nodes": [], "edges": []}

    # 시각화를 위한 서브그래프 생성 (top 150)
    H = _subgraph_for_viz(edges, pr_df, top_n=150, degree_min=1)
    if H.number_of_nodes() == 0:
        return {"nodes": [], "edges": []}

    pr_map = pr_df.set_index("title")["pagerank"] if pr_df is not None and not pr_df.empty else pd.Series(dtype=float)
    comm = _compute_communities(H)

    # PR 기반 사이즈 10~40로 스케일
    pr_vals = {n: float(pr_map.get(n, 0.0)) for n in H.nodes}
    size_map = {n: 15.0 for n in H.nodes}
    if pr_vals:
        vmin, vmax = min(pr_vals.values()), max(pr_vals.values())
        if vmax > vmin:
            size_map = {n: 10.0 + 30.0 * ((v - vmin) / (vmax - vmin)) for n, v in pr_vals.items()}

    nodes_out = [
        {"id": n, "group": comm.get(n, 0), "size": size_map.get(n, 15.0)}
        for n in H.nodes
    ]
    edges_out = [
        {"source": u, "target": v, "weight": float(d.get("weight", 1.0))}
        for u, v, d in H.edges(data=True)
    ]

    return {"nodes": nodes_out, "edges": edges_out}


# pipeline.py (약 837번째 줄)

def format_item_4_indicators(stats_df: pd.DataFrame, top_n: int = 20):
    """유망성 지표 상위 N개를 JSON 목록으로 반환합니다."""
    if stats_df.empty:
        return []

    INDICATORS = ["공급부상성", "수요부상성", "확산성", "기술집약도"]
    stats_scored = score_1to5(stats_df, INDICATORS)

    # '기술집약도' 기준으로 정렬하여 상위 top_n개 반환
    ranked = stats_scored.sort_values("기술집약도", ascending=False)

    # --- [수정된 부분] ---
    # .fillna(None) 대신, .where()를 사용하여 모든 NaN/NaT 값을 None으로 안전하게 변환
    df_out = ranked.head(top_n).copy()

    # DataFrame을 object 타입으로 변경해야 datetime, str, float이 섞여도 .where()가 작동함
    df_out = df_out.astype(object).where(pd.notnull(df_out), None)

    return df_out.to_dict('records')
    # --- [여기까지 수정] ---


def format_item_5_trends(seed_title: str, edit_wide_path: str, view_wide_path: str):
    """연도별 추세 데이터를 JSON으로 반환합니다."""
    try:
        edit_wide = _read_excel(edit_wide_path)
        view_wide = _read_excel(view_wide_path)

        s_ed = extract_year_series(edit_wide, seed_title)
        s_vw = extract_year_series(view_wide, seed_title)

        if s_ed.empty and s_vw.empty:
            return {"years": [], "edits": [], "views": []}

        if not s_ed.empty and not s_vw.empty:
            years_set = set(s_ed.index) & set(s_vw.index)
        else:
            years_set = set(s_ed.index) or set(s_vw.index)

        years = sorted([y for y in years_set if y > 0])  # 0년 제외
        if not years:
            return {"years": [], "edits": [], "views": []}

        ed_vals = s_ed.reindex(years).fillna(0.0).to_numpy().ravel().tolist() if not s_ed.empty else [0.0] * len(years)
        vw_vals = s_vw.reindex(years).fillna(0.0).to_numpy().ravel().tolist() if not s_vw.empty else [0.0] * len(years)

        return {"years": years, "edits": ed_vals, "views": vw_vals}

    except Exception as e:
        print(f"[{seed_title}] 추세 데이터 생성 오류: {e}")
        return {"years": [], "edits": [], "views": []}


# --- API가 호출할 메인 함수 ---

def get_analysis_results(seed: str, n_depth: int = 1):
    """
    1. 파이프라인을 실행(또는 캐시 로드)
    2. 결과를 읽어옴
    3. 5가지 항목으로 포맷팅하여 반환
    (*** 에러 추적을 위해 try...except 블록 추가됨 ***)
    """

    try:
        # 1. 파이프라인 실행
        print("[API 포맷팅] 1. 파이프라인 실행 시작...")
        paths, outs, seed_true, seed_url, seed_summary = run_analysis_pipeline(
            seed=seed,
            n_depth=n_depth,
            use_existing=True
        )
        print("[API 포맷팅] 1. 파이프라인 실행 완료.")

        # 2. 결과 파일 로드
        print("[API 포맷팅] 2. 결과 엑셀 로드 중...")
        pr_df = _read_excel(paths["pagerank"])
        stats_df = _read_excel(paths["stats"])
        print("[API 포맷팅] 2. 결과 엑셀 로드 완료.")

        # 3. 5가지 항목으로 포맷팅
        output = {}

        print("[API 포맷팅] 3-1. (Entry) 포맷팅 중...")
        output["item_1_entry"] = format_item_1_entry(seed_true, seed_url)

        print("[API 포맷팅] 3-2. (Summary) 포맷팅 중...")
        output["item_2_summary"] = format_item_2_summary(seed_summary)

        print("[API 포맷팅] 3-3. (Network) 포맷팅 중...")
        output["item_3_network"] = format_item_3_network(paths["filter"], pr_df)

        print("[API 포맷팅] 3-4. (Indicators) 포맷팅 중...")
        output["item_4_indicators"] = format_item_4_indicators(stats_df, top_n=20)

        print("[API 포맷팅] 3-5. (Trends) 포맷팅 중...")
        output["item_5_trends"] = format_item_5_trends(
            seed_true,
            outs.get("edit", paths["edit"]),
            outs.get("pageviews", paths["pageviews"])
        )

        print("[API 포맷팅] 3. 모든 포맷팅 완료. 결과 반환.")
        return output

    except ValueError as ve:
        # 400 Bad Request의 원인
        print(f"\n[API 포맷팅 ERROR] ValueError 발생: {ve}")
        print("--- 엑셀 파일 데이터 포맷팅 중 오류가 발생했습니다 ---")
        # api.py가 이 에러를 잡아 400을 반환합니다.
        raise ve

    except Exception as e:
        # 500 Internal Error의 원인
        print(f"\n[API 포맷팅 CRITICAL] 예상치 못한 에러: {e}")
        # api.py가 이 에러를 잡아 500을 반환합니다.
        raise e


# ===============================================
# 멀티 시드 분석 파이프라인
# ===============================================

def merge_filtered_edges(
    filter_xlsx_list: List[str],
    out_xlsx: str,
    seed_nodes: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    여러 시드의 필터링된 엣지 리스트를 하나로 병합.
    겹치는 엣지는 가중치 합산 (교차 도메인 중요도 반영).
    seed_nodes가 주어지면, 엣지가 없는 시드도 고립 노드로 추가.
    """
    if exists_nonempty(out_xlsx):
        return pd.read_excel(out_xlsx)

    frames = []
    for xlsx in filter_xlsx_list:
        if exists_nonempty(xlsx, min_bytes=1):
            try:
                edges = load_edges(xlsx)
                if not edges.empty:
                    frames.append(edges)
            except Exception as e:
                print(f"[merge] 엣지 로드 실패 ({xlsx}): {e}")

    if not frames:
        merged = pd.DataFrame(columns=["From_title", "To_seealso", "weight"])
    else:
        merged = pd.concat(frames, ignore_index=True)
        merged["From_title"] = merged["From_title"].map(title_space)
        merged["To_seealso"] = merged["To_seealso"].map(title_space)
        merged = merged[merged["From_title"] != merged["To_seealso"]]
        merged = merged.groupby(["From_title", "To_seealso"])["weight"].sum().reset_index()

    # 엣지가 없는 시드도 네트워크에 고립 노드로 포함
    if seed_nodes:
        existing_nodes = set(merged["From_title"].tolist() + merged["To_seealso"].tolist())
        for sn in seed_nodes:
            sn_norm = title_space(sn)
            if sn_norm not in existing_nodes:
                print(f"[merge] '{sn_norm}' 엣지 없음 → 고립 노드로 추가")
                # 자기 자신을 가리키는 엣지 대신, 다른 시드와 연결
                # (고립 노드를 PageRank에 포함시키기 위한 최소 엣지)
                if not merged.empty:
                    # 기존 노드 중 하나와 가중치 0.1로 연결
                    anchor = merged["From_title"].iloc[0]
                    merged = pd.concat([merged, pd.DataFrame([{
                        "From_title": sn_norm,
                        "To_seealso": anchor,
                        "weight": 0.1,
                    }])], ignore_index=True)
                else:
                    # 모든 시드가 엣지 없는 경우: 시드 간 연결
                    merged = pd.concat([merged, pd.DataFrame([{
                        "From_title": sn_norm,
                        "To_seealso": sn_norm,
                        "weight": 0.0,
                    }])], ignore_index=True)

    merged.to_excel(out_xlsx, index=False)
    return merged


def merge_xtools_data(per_seed_outs: List[Dict[str, str]], out_dir: str) -> Dict[str, str]:
    """
    시드별 _all_edit/info/link/pageviews.xlsx를 통합.
    - edit/pageviews: groupby(title) → 연도별 max
    - info/link: drop_duplicates(title)
    """
    ensure_dir(out_dir)
    result = {}

    for data_type in ["edit", "info", "link", "pageviews"]:
        out_path = os.path.join(out_dir, f"merged_all_{data_type}.xlsx")
        # 입력 파일 중 하나라도 merged 파일보다 새로우면 재생성
        _input_newer = False
        if exists_nonempty(out_path):
            out_mtime = os.path.getmtime(out_path)
            for outs in per_seed_outs:
                p = outs.get(data_type, "")
                try:
                    if p and os.path.isfile(p) and os.path.getmtime(p) > out_mtime:
                        _input_newer = True
                        break
                except Exception:
                    pass
        if exists_nonempty(out_path) and not _input_newer:
            result[data_type] = out_path
            continue

        frames = []
        for outs in per_seed_outs:
            path = outs.get(data_type, "")
            if path and exists_nonempty(path, min_bytes=1):
                try:
                    df = pd.read_excel(path)
                    if not df.empty:
                        frames.append(df)
                except Exception as e:
                    print(f"[merge_xtools] {data_type} 로드 실패 ({path}): {e}")

        if frames:
            merged = pd.concat(frames, ignore_index=True)
            if "title" in merged.columns:
                merged["title"] = merged["title"].map(title_space)
                if data_type in ("edit", "pageviews"):
                    year_cols = [c for c in merged.columns if re.fullmatch(r"\d{4}", str(c))]
                    if year_cols:
                        merged = merged.groupby("title")[year_cols].max().reset_index()
                else:
                    merged = merged.drop_duplicates(subset=["title"], keep="first")
            merged.to_excel(out_path, index=False)
        else:
            pd.DataFrame(columns=["title"]).to_excel(out_path, index=False)

        result[data_type] = out_path

    return result


def run_multi_seed_pipeline(
    seeds: List[str],
    n_depth: int,
    use_existing: bool = True,
):
    """
    멀티 시드 분석 파이프라인.
    1) 시드별 기존 run_analysis_pipeline() 순차 호출
    2) 필터링된 엣지 병합
    3) XTools 데이터 병합
    4) 통합 PageRank
    5) 통합 통계
    반환: (multi_paths, per_seed_results, merged_outs)
    """
    if not seeds or all(not s.strip() for s in seeds):
        raise ValueError("시드를 1개 이상 입력하세요.")

    seeds = [s.strip() for s in seeds if s.strip()]

    # 멀티 시드 출력 디렉토리
    run_hash = multi_seed_hash(seeds, n_depth)
    multi_dir = Path(BASE_RUN_DIR) / f"multi_{run_hash}"
    merged_xtools_dir = multi_dir / "merged_xtools"
    ensure_dir(multi_dir)
    ensure_dir(merged_xtools_dir)

    multi_paths = {
        "merged_edges": str(multi_dir / "merged_edges.xlsx"),
        "merged_pagerank": str(multi_dir / "merged_pagerank.xlsx"),
        "merged_stats": str(multi_dir / "merged_statistics.xlsx"),
        "merged_xtools_dir": str(merged_xtools_dir),
        "manifest": str(multi_dir / "manifest.json"),
    }

    # --- Phase 1: 시드별 파이프라인 ---
    per_seed_results = []
    failed_seeds = []

    for i, seed in enumerate(seeds):
        try:
            # 시드 간 rate limit 방지
            if i > 0:
                import time
                time.sleep(2)
            print(f"[multi] '{seed}' 파이프라인 실행... ({i+1}/{len(seeds)})")
            result = run_analysis_pipeline(seed, n_depth, use_existing)
            per_seed_results.append(result)
        except Exception as e:
            print(f"[multi] '{seed}' 실패: {e}")
            failed_seeds.append((seed, str(e)))

    if not per_seed_results:
        raise RuntimeError(f"모든 시드가 실패했습니다: {failed_seeds}")

    # --- Phase 2: 엣지 병합 (시드 노드 강제 포함) ---
    print("[multi] 필터링된 네트워크 병합 중...")
    filter_files = [r[0]["filter"] for r in per_seed_results]
    seed_true_names = [r[2] for r in per_seed_results]  # 각 시드의 True Title
    merge_filtered_edges(filter_files, multi_paths["merged_edges"], seed_nodes=seed_true_names)

    # --- Phase 3: XTools 병합 ---
    print("[multi] XTools 데이터 병합 중...")
    per_seed_outs = [r[1] for r in per_seed_results]
    merged_outs = merge_xtools_data(per_seed_outs, str(merged_xtools_dir))

    # --- Phase 4: 통합 PageRank ---
    print("[multi] 통합 PageRank 계산 중...")
    pr_df = compute_pagerank(multi_paths["merged_edges"], multi_paths["merged_pagerank"],
                             seed_nodes=seed_true_names)

    # --- Phase 5: 통합 통계 ---
    print("[multi] 통합 통계 산출 중...")
    compute_statistics(
        edit_xlsx=merged_outs["edit"],
        pageviews_xlsx=merged_outs["pageviews"],
        link_xlsx=merged_outs["link"],
        info_xlsx=merged_outs["info"],
        pr_df=pr_df,
        out_xlsx=multi_paths["merged_stats"],
        ref_year=None,
    )

    # --- manifest 저장 ---
    manifest = {
        "seeds": seeds,
        "n_depth": n_depth,
        "created_at": datetime.datetime.now().isoformat(),
        "per_seed_dirs": [slugify_seed(r[2]) for r in per_seed_results],
        "failed_seeds": failed_seeds,
    }
    Path(multi_paths["manifest"]).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"[multi] 완료. 결과: {multi_dir}")
    return multi_paths, per_seed_results, merged_outs


def get_multi_analysis_results(seeds: List[str], n_depth: int = 1) -> dict:
    """
    멀티 시드 API 응답 생성.
    """
    try:
        multi_paths, per_seed_results, merged_outs = run_multi_seed_pipeline(
            seeds=seeds, n_depth=n_depth, use_existing=True
        )

        pr_df = _read_excel(multi_paths["merged_pagerank"])
        stats_df = _read_excel(multi_paths["merged_stats"])

        output = {}

        # 1. 시드별 진입 아이템
        output["item_1_entries"] = [
            format_item_1_entry(r[2], r[3]) for r in per_seed_results
        ]

        # 2. 시드별 요약
        output["item_2_summaries"] = [
            format_item_2_summary(r[4]) for r in per_seed_results
        ]

        # 3. 통합 네트워크
        output["item_3_network"] = format_item_3_network(
            multi_paths["merged_edges"], pr_df
        )

        # 4. 통합 유망성 지표
        output["item_4_indicators"] = format_item_4_indicators(stats_df, top_n=20)

        # 5. 시드별 트렌드
        trends = {}
        for r in per_seed_results:
            paths, outs, seed_true, _, _ = r
            trends[seed_true] = format_item_5_trends(
                seed_true,
                outs.get("edit", paths["edit"]),
                outs.get("pageviews", paths["pageviews"]),
            )
        output["item_5_trends"] = trends

        # 메타 정보
        output["meta"] = {
            "seed_count": len(per_seed_results),
            "seeds": [r[2] for r in per_seed_results],
            "n_depth": n_depth,
        }

        return output

    except ValueError as ve:
        raise ve
    except Exception as e:
        raise e