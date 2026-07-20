
# app.py (patched)
# -*- coding: utf-8 -*-

from __future__ import annotations
import os
import re
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

# ====== 외부 모듈 (같은 폴더에 있어야 함) ======
import wiki_crawling
import streamlit.components.v1 as components
from matplotlib import colors as mcolors

# =====================
# 기본/글꼴/레이아웃 설정
# =====================
st.set_page_config(page_title="유망아이템 분석도구", page_icon="🧭", layout="wide")

def _pick_korean_font() -> Optional[str]:
    from matplotlib import font_manager as fm
    candidates = [
        "AppleGothic",
        "NanumGothic",
        "NanumBarunGothic",
        "Malgun Gothic",
        "Noto Sans CJK KR",
        "Noto Sans KR",
    ]
    try:
        avail = {f.name for f in fm.fontManager.ttflist}
    except Exception:
        avail = set()
    for name in candidates:
        if name in avail:
            return name
    file_candidates = [
        "/System/Library/Fonts/AppleGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "C:\\Windows\\Fonts\\malgun.ttf",
    ]
    for fp in file_candidates:
        if os.path.exists(fp):
            try:
                return matplotlib.font_manager.FontProperties(fname=fp).get_name()
            except Exception:
                pass
    return None

_font = _pick_korean_font()
if _font:
    matplotlib.rcParams["font.family"] = _font
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["figure.dpi"] = 120

# 시각적 여백(너무 꽉 차는 느낌 방지)
st.markdown(
    """
    <style>
      .block-container { max-width: 1100px; padding-top: 1rem; padding-bottom: 2rem; }
      .stMarkdown, .stDataFrame, .stPlotlyChart, .stImage { margin-bottom: 1rem; }
    </style>
    """,
    unsafe_allow_html=True
)


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
    return re.sub(r"\s+", " ", str(s).replace("_", " ").strip())


def listdir_nods(path: str | Path) -> List[str]:
    p = Path(path)
    if not p.exists() or not p.is_dir():
        return []
    return [f for f in os.listdir(p) if not f.startswith(".")]


# ======================
# 1) True title 확인/저장
# ======================
@st.cache_data(show_spinner=False)
def resolve_true_title(seed: str, out_xlsx: str) -> pd.DataFrame:
    if exists_nonempty(out_xlsx):
        df = pd.read_excel(out_xlsx)
    else:
        rows = wiki_crawling.check_seed([seed])
        df = pd.DataFrame(rows, columns=["No", "Title", "True_title", "Category", "Contents", "URL"])
        df.to_excel(out_xlsx, index=False)
    df["Title"] = df["Title"].map(title_space)
    df["True_title"] = df["True_title"].map(title_space)
    return df


# ============================
# 2) 확장/필터 (심플 스킵 방식)
# ============================

def expand_network(seed_true_title: str, out_xlsx: str, n: int) -> str:
    """
    확장 결과 파일의 '실제 경로'를 문자열로 반환.
    - 기본 경로 runs/<seed>/seed_item/<n>차시 확장 최종_결과.xlsx
    - 생성기가 True_title 하위 폴더에 쓸 때를 대비해 대체 경로도 탐색
    """
    if exists_nonempty(out_xlsx):
        return out_xlsx

    out_dir = str(Path(out_xlsx).parent)
    ensure_dir(out_dir)

    with st.spinner("위키 네트워크 확장 중…"):
        wiki_crawling.n_char_crawler([seed_true_title], n, out_dir)

    # 1) 기본 경로
    if exists_nonempty(out_xlsx):
        return out_xlsx

    # 2) 대체 경로: runs/<seed>/seed_item/<True_title>/<n>차시 확장 최종_결과.xlsx
    alt = Path(out_dir) / seed_true_title / f"{n}차시 확장 최종_결과.xlsx"
    if exists_nonempty(alt):
        return str(alt)

    # 3) 그래도 없으면 에러
    raise FileNotFoundError(
        f"확장 결과 파일을 찾을 수 없습니다:\n - {out_xlsx}\n - {alt}"
    )


@st.cache_data(show_spinner=False)
def _read_excel(path: str) -> pd.DataFrame:
    return pd.read_excel(path) if exists_nonempty(path, 1) else pd.DataFrame()


def filter_network(
    src_xlsx: str,
    seed_true_title: str,
    out_xlsx: str,
    rule_file: str = "./data/wiki_rule.xlsx",
    mode: str = "balanced",  # 인터페이스 유지용
) -> str:
    """
    wiki_rule.xlsx의 title/category 룰에 따라 네트워크를 필터링한다.
    - To_seealso가 title 룰에 매칭되면 제외 후보
    - Category가 category 룰에 매칭되면 해당 행의 To_seealso도 제외 후보
    - From_title / To_seealso가 제외 후보에 포함된 행은 제거
    - seed_true_title은 화이트리스트로 보존
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
            st.warning(f"룰 파일 로드 오류: {e} → 룰 생략")
    else:
        if mode != "off":
            st.warning(f"룰 파일을 찾을 수 없어({rule_file}) 룰 생략")

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
        st.warning(f"필터로 {drop_ratio:.1%} 제거 감지 → 룰 스킵")
        keep_mask[:] = True

    result = df[keep_mask].copy()

    # 자기루프 제거 + 중복 통합
    result = result[result[col_from] != result[col_to]]
    result = result.groupby([col_from, col_to]).size().reset_index(name="weight")

    # 저장
    result.rename(columns={col_from: "From_title", col_to: "To_seealso"}, inplace=True)
    result.to_excel(out_xlsx, index=False)

    with st.expander("🔎 필터링 요약", expanded=False):
        st.write(f"원본: {len(df):,} → 필터 후: {len(result):,} "
                 f"(잔존 {(len(result) / max(len(df), 1)) * 100:.1f}%)")
        if col_cat:
            st.write("Category 상위:", df[col_cat].astype(str).str.lower().value_counts().head(10))

    return out_xlsx


# =========================================
# 3) Xtools 수집(센티넬 파일로 스킵) + 통합
# =========================================

def xtools_collect(seed_true_title: str, filtered_xlsx: str, xtools_dir: str, flag_file: str):
    """수집은 파일 여러 개를 만들므로 센티넬(.collect_done)로 스킵."""
    if exists_nonempty(flag_file, min_bytes=1):
        return
    ensure_dir(xtools_dir)
    df = pd.read_excel(filtered_xlsx)
    nodes = set(map(title_space, list(df.get("From_title", [])) + list(df.get("To_seealso", []))))
    nodes = [n for n in nodes if str(n).strip()]
    if nodes:
        with st.spinner("편집/정보/링크/조회수 수집 중…"):
            wiki_crawling.wiki_info_crawl(nodes, xtools_dir)
    Path(flag_file).write_text("done", encoding="utf-8")


def xtools_integrate(seed_true_title: str, xtools_dir: str) -> Dict[str, str]:
    """
    편집/조회/링크/정보 '원본(row)'과 '통합(wide)' 포맷 혼재를 흡수.
    _all_*.xlsx 통합본은 입력으로 스킵.
    """
    files = listdir_nods(xtools_dir)
    edit_pat = re.compile(r"(?<!_all)_edit\.xlsx$")
    info_pat = re.compile(r"(?<!_all)_info\.xlsx$")
    link_pat = re.compile(r"(?<!_all)_link\.xlsx$")
    view_pat = re.compile(r"(?<!_all)_pageviews\.xlsx$")

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

        # long → pivot
        if ("edits" in df.columns) and (("year" in df.columns) or ("timestamp" in df.columns)):
            if "year" not in df.columns and "timestamp" in df.columns:
                df["year"] = df["timestamp"].astype(str).str[:4]
            df["year"] = df["year"].astype(str)
            piv = df.pivot(index="title", columns="year", values="edits")
            edit_frames.append(piv)
            continue

        # wide(이미 연도 가로열)
        year_cols = [str(c) for c in df.columns if re.fullmatch(r"\d{4}", str(c))]
        year_cols = [c for c in year_cols if 2000 <= int(c) <= 2100]
        if year_cols:
            edit_frames.append(df[["title"] + year_cols].set_index("title"))

    edit_out = os.path.join(xtools_dir, f"{seed_true_title}_all_edit.xlsx")
    if not exists_nonempty(edit_out):
        if edit_frames:
            e = pd.concat(edit_frames, axis=0, ignore_index=False)
            e = e.groupby(e.index).max()
            e = e.reset_index()
        else:
            e = pd.DataFrame(columns=["title"])
        e.sort_values("title").to_excel(edit_out, index=False)

    # ---- info
    info_out = os.path.join(xtools_dir, f"{seed_true_title}_all_info.xlsx")
    if not exists_nonempty(info_out):
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
    if not exists_nonempty(link_out):
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
    if not exists_nonempty(view_out):
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
# 3-b) 네트워크 고급 시각화 유틸
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
        from networkx.algorithms.community import greedy_modularity_communities

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


def draw_network_advanced(
    filtered_xlsx: str,
    pr_df: pd.DataFrame,
    mode: str = "커뮤니티(색상)",
    top_n: int = 150,
    degree_min: int = 1,
    label_top: int = 20,
):
    edges = load_edges(filtered_xlsx)
    if edges.empty:
        st.info("네트워크가 비어 있습니다.")
        return

    H = _subgraph_for_viz(edges, pr_df, top_n=top_n, degree_min=degree_min)
    if H.number_of_nodes() == 0:
        st.info("표시할 노드가 없습니다. 필터를 완화해 보세요.")
        return

    pr_map = pr_df.set_index("title")["pagerank"] if pr_df is not None and not pr_df.empty else pd.Series(dtype=float)

    # 커뮤니티 색상
    if mode == "커뮤니티(색상)":
        comm = _compute_communities(H)
    else:
        comm = {n: 0 for n in H.nodes}
    palette = list(plt.cm.tab20.colors)
    node_colors = [palette[comm[n] % len(palette)] for n in H.nodes]

    # 노드 크기 (PR 기반)
    sizes = np.array([float(pr_map.get(n, 0.0)) for n in H.nodes])
    if sizes.size and sizes.max() > 0:
        sizes = 80 + 1700 * (sizes / sizes.max())
    else:
        sizes = np.full(H.number_of_nodes(), 120.0)

    # 엣지 굵기 (가중치 정규화)
    w = np.array([float(H[u][v].get("weight", 1.0)) for u, v in H.edges])
    if w.size:
        w = (w - w.min()) / (w.ptp() if w.ptp() > 0 else 1)
        edge_w = 0.3 + 1.7 * w
    else:
        edge_w = np.full(len(H.edges), 0.6)

    pos = nx.spring_layout(H, k=0.35, seed=42, weight="weight")

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    nx.draw_networkx_edges(H, pos, arrows=False, width=edge_w, alpha=0.25)
    nx.draw_networkx_nodes(H, pos, node_size=sizes, node_color=node_colors, alpha=0.9, linewidths=0.2, edgecolors="#666")

    # 라벨: PR 상위 label_top
    nodes_sorted = sorted(H.nodes, key=lambda n: float(pr_map.get(n, 0.0)), reverse=True)
    label_nodes = set(nodes_sorted[: int(label_top)])
    labels = {n: n for n in H.nodes if n in label_nodes}
    nx.draw_networkx_labels(H, pos, labels=labels, font_size=8)

    ax.set_axis_off()
    st.pyplot(fig, clear_figure=True)


def render_pyvis_network(
    filtered_xlsx: str,
    pr_df: pd.DataFrame,
    top_n: int = 150,
    degree_min: int = 1,
) -> Optional[str]:
    """PyVis로 인터랙티브 HTML을 생성하고 경로를 반환합니다."""
    try:
        from pyvis.network import Network
    except Exception:
        st.error("pyvis가 설치되어 있지 않습니다. `pip install pyvis` 후 다시 시도하세요.")
        return None

    import time
    edges = load_edges(filtered_xlsx)
    if edges.empty:
        st.info("네트워크가 비어 있습니다.")
        return None

    H = _subgraph_for_viz(edges, pr_df, top_n=top_n, degree_min=degree_min)
    if H.number_of_nodes() == 0:
        st.info("표시할 노드가 없습니다. 필터를 완화해 보세요.")
        return None

    pr_map = pr_df.set_index("title")["pagerank"] if pr_df is not None and not pr_df.empty else pd.Series(dtype=float)

    # 커뮤니티 색상
    comm = _compute_communities(H)
    palette = list(plt.cm.tab20.colors)

    # PR 기반 사이즈 10~40로 스케일
    pr_vals = {n: float(pr_map.get(n, 0.0)) for n in H.nodes}
    if pr_vals:
        vmin, vmax = min(pr_vals.values()), max(pr_vals.values())
        def scale(v: float, lo=10.0, hi=40.0):
            return lo if vmax <= vmin else lo + (hi - lo) * ((v - vmin) / (vmax - vmin))
        size_map = {n: scale(v) for n, v in pr_vals.items()}
    else:
        size_map = {n: 15.0 for n in H.nodes}

    net = Network(height="620px", width="100%", bgcolor="#ffffff", font_color="#111111", directed=False)
    net.barnes_hut(gravity=-2000, central_gravity=0.2, spring_length=120, spring_strength=0.01, damping=0.09)

    # 노드 추가
    for n in H.nodes:
        color = mcolors.to_hex(palette[comm[n] % len(palette)])
        title = f"<b>{n}</b><br>PR: {pr_vals.get(n, 0.0):.5f}<br>degree: {H.degree(n)}"
        net.add_node(n, label=n, title=title, value=float(pr_vals.get(n, 0.0)), color=color, size=size_map.get(n, 12.0))

    # 엣지 추가
    for u, v, d in H.edges(data=True):
        w = float(d.get("weight", 1.0))
        net.add_edge(u, v, value=w, opacity=0.25)

    html_path = str(Path(filtered_xlsx).parent / f"network_interactive_{int(time.time())}.html")
    # (기존)
    try:
        net.show(html_path)
    except Exception:
        net.save_graph(html_path)

    # (변경)
    net.save_graph(html_path)
    return html_path


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


@st.cache_data(show_spinner=False)
def compute_pagerank(filtered_xlsx: str, out_xlsx: str) -> pd.DataFrame:
    if exists_nonempty(out_xlsx):
        return pd.read_excel(out_xlsx)
    edges = load_edges(filtered_xlsx)
    if edges.empty:
        pr_df = pd.DataFrame(columns=["title", "pagerank"])
        pr_df.to_excel(out_xlsx, index=False)
        return pr_df
    G = nx.from_pandas_edgelist(edges, "From_title", "To_seealso", edge_attr="weight", create_using=nx.DiGraph())
    try:
        pr = nx.pagerank(G, alpha=0.85, max_iter=200, weight="weight")
    except nx.PowerIterationFailedConvergence:
        pr = nx.pagerank(G, alpha=0.80, max_iter=400, weight="weight", tol=1e-6)
    pr_df = (
        pd.DataFrame.from_dict(pr, orient="index", columns=["pagerank"]).reset_index().rename(columns={"index": "title"})
    )
    pr_df = pr_df.sort_values("pagerank", ascending=False).reset_index(drop=True)
    pr_df.to_excel(out_xlsx, index=False)
    return pr_df


def draw_network(filtered_xlsx: str, pr_df: pd.DataFrame, max_nodes: int = 120, label_top: int = 15):
    edges = load_edges(filtered_xlsx)
    if edges.empty:
        st.info("네트워크가 비어 있습니다.")
        return
    keep_nodes = set(pr_df["title"].head(max_nodes).tolist()) if not pr_df.empty else \
                 set(edges["From_title"]).union(edges["To_seealso"])
    sub = edges[edges["From_title"].isin(keep_nodes) | edges["To_seealso"].isin(keep_nodes)]
    if sub.empty:
        sub = edges.head(min(1000, len(edges)))
    G = nx.from_pandas_edgelist(sub, "From_title", "To_seealso", edge_attr="weight", create_using=nx.DiGraph())
    pr_map = pr_df.set_index("title")["pagerank"] if not pr_df.empty else pd.Series(dtype=float)
    sizes = [50 + 1450 * float(pr_map.get(n, 0.0)) for n in G.nodes]
    pos = nx.spring_layout(G, k=0.35, seed=42)
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    nx.draw_networkx_edges(G, pos, arrows=False, width=0.6, alpha=0.35)
    nx.draw_networkx_nodes(G, pos, node_size=sizes, alpha=0.85, linewidths=0.2, edgecolors="#666666")
    top_nodes = set(pr_df.head(label_top)["title"].tolist()) if not pr_df.empty else set()
    labels = {n: n for n in G.nodes if n in top_nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
    ax.set_axis_off()
    st.pyplot(fig, clear_figure=True)


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
    if exists_nonempty(out_xlsx):
        return pd.read_excel(out_xlsx)

    # --- edit
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

    # --- pageviews
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

    # --- link
    link_df = _read_excel(link_xlsx)
    if not link_df.empty and "title" in link_df.columns:
        if "links_in_count" in link_df.columns:
            link_df = link_df[["title", "links_in_count"]].rename(columns={"links_in_count": "확산성"})
        elif "links_in" in link_df.columns:
            link_df = link_df[["title", "links_in"]].rename(columns={"links_in": "확산성"})
        else:
            link_df = link_df[["title"]].copy(); link_df["확산성"] = 0
    else:
        link_df = pd.DataFrame(columns=["title", "확산성"])  # 안전

    # --- info
    info_df = _read_excel(info_xlsx)
    if not info_df.empty and "title" in info_df.columns:
        col_map = {}
        if "created_at" in info_df.columns: col_map["created_at"] = "생성일"
        if "modified_at" in info_df.columns: col_map["modified_at"] = "최근편집일"
        if col_map:
            info_df = info_df[["title"] + list(col_map.keys())].rename(columns=col_map)
        else:
            info_df = info_df[["title"]].copy(); info_df["생성일"] = ""; info_df["최근편집일"] = ""
    else:
        info_df = pd.DataFrame(columns=["title", "생성일", "최근편집일"])  # 안전

    from functools import reduce
    frames = [
        edit_df[["title", "공급부상성", "총편집수"]] if not edit_df.empty else pd.DataFrame(columns=["title", "공급부상성", "총편집수"]),
        pv_df[["title", "수요부상성", "총조회수"]] if not pv_df.empty else pd.DataFrame(columns=["title", "수요부상성", "총조회수"]),
        link_df[["title", "확산성"]] if not link_df.empty else pd.DataFrame(columns=["title", "확산성"]),
        info_df[["title", "생성일", "최근편집일"]] if not info_df.empty else pd.DataFrame(columns=["title", "생성일", "최근편집일"]),
    ]

    # inner merge 대신 outer→누락 보호 후 fillna(0) 처리
    core = reduce(lambda x, y: pd.merge(x, y, on="title", how="outer"), frames)

    pr_use = pr_df.rename(columns={"pagerank": "기술집약도"}) if not pr_df.empty else pd.DataFrame(columns=["title", "기술집약도"])  # type: ignore
    stats = pd.merge(core, pr_use[["title", "기술집약도"]] if not pr_use.empty else pd.DataFrame(columns=["title", "기술집약도"]),
                     on="title", how="outer").fillna(0)
    stats = stats[["title", "공급부상성", "수요부상성", "확산성", "기술집약도", "총편집수", "총조회수", "생성일", "최근편집일"]]
    stats.to_excel(out_xlsx, index=False)
    return stats


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


# ================
# 레이더(멀티) 차트
# ================

def radar_multi_by_items(
    df_scores: pd.DataFrame,
    items: list[str],
    indicators: list[str],
    title: str = "아이템군 분석 - 부상성 그래프",
    prefer_scores: bool = True,   # *_score가 있으면 우선 사용
):
    if len(items) < 3:
        st.info("레이더 차트는 최소 3개 이상의 아이템 필요")
        return

    data = df_scores[df_scores["title"].isin(items)].set_index("title").reindex(items)

    # *_score 존재하면 그걸 쓰고, 없으면 원본 사용
    cols_to_plot, legend_labels = [], []
    for c in indicators:
        sc = f"{c}_score"
        if prefer_scores and sc in data.columns:
            cols_to_plot.append(sc)
            legend_labels.append(c)   # 범례는 원래 이름
        else:
            cols_to_plot.append(c)
            legend_labels.append(c)

    n = len(items)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # 축 라벨 = 아이템
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(items, fontsize=9, fontweight="bold")

    # y축 범위 설정: *_score를 쓰면 1~5, 원본이면 자동
    using_scores = all(col.endswith("_score") for col in cols_to_plot)
    if using_scores:
        # y축 범위를 1~6으로 잡아 여유 공간 확보
        ax.set_ylim(1, 6)
        ax.set_rgrids([1, 2, 3, 4, 5], angle=90, fontsize=8, color="gray", alpha=0.6)
    else:
        ymax = float(pd.DataFrame({c: data[c] for c in cols_to_plot}).max().max()) if len(data) else 1
        ax.set_ylim(0, max(1.0, ymax * 1.1))

    colors = plt.cm.Set2.colors
    for i, (coln, label) in enumerate(zip(cols_to_plot, legend_labels)):
        vals = pd.to_numeric(data[coln], errors="coerce").fillna(np.nan).tolist()
        vals = [0 if (np.isnan(v) and not using_scores) else (1 if (np.isnan(v) and using_scores) else v) for v in vals]
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=2, label=label, color=colors[i % len(colors)])
        ax.fill(angles, vals, alpha=0.12, color=colors[i % len(colors)])

    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.22, 1.08))
    st.pyplot(fig, clear_figure=True)


# ===============
# 연도별 추세 라인차트 (seed 중심)
# ===============

def plot_yearly_trends(seed_title: str, edit_wide_path: str, view_wide_path: str):
    st.subheader(f"📈 {seed_title} 연도별 활동 추세")

    try:
        edit_wide = _read_excel(edit_wide_path)
        view_wide = _read_excel(view_wide_path)

        def extract_year_series(df: pd.DataFrame, title: str) -> pd.Series:
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

        s_ed = extract_year_series(edit_wide, seed_title)
        s_vw = extract_year_series(view_wide, seed_title)

        if s_ed.empty and s_vw.empty:
            st.info("연도별 편집/조회 데이터가 없습니다.")
            return

        if not s_ed.empty and not s_vw.empty:
            years = sorted(set(s_ed.index) & set(s_vw.index))
        else:
            years = sorted(set(s_ed.index) or set(s_vw.index))
        if len(years) == 0:
            st.info("겹치는 연도가 없어 추세를 그릴 수 없습니다.")
            return

        ed_vals = s_ed.reindex(years).to_numpy().ravel() if not s_ed.empty else np.zeros(len(years))
        vw_vals = s_vw.reindex(years).to_numpy().ravel() if not s_vw.empty else np.zeros(len(years))

        fig, ax1 = plt.subplots(figsize=(7, 4))
        ax1.plot(years, ed_vals, label="편집수")
        ax1.set_xlabel("연도")
        ax1.set_ylabel("편집수", color="C0")

        ax2 = ax1.twinx()
        ax2.plot(years, vw_vals, label="조회수", color="C2")
        ax2.set_ylabel("조회수", color="C2")

        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc="upper left")
        st.pyplot(fig, clear_figure=True)
    except Exception as e:
        st.warning(f"추세 그래프 생성 중 오류: {e}")


# =========================
# Streamlit 입력/실행/보고서
# =========================

st.title("🧭 유망아이템 단일 시드 분석")

# --- 입력 ---
seed = st.text_input("시드 입력", placeholder="예: Nvidia", value=st.session_state.get("seed", ""))
n_depth = st.number_input("확장 차수 (n)", min_value=1, max_value=3, value=st.session_state.get("n_depth", 1), step=1)
use_existing = st.toggle("이미 생성된 엑셀 산출물이 있으면 그대로 사용", value=True)
run = st.button("전체 실행")

# --- 실행(버튼 눌렀을 때만 파이프라인) ---
if run:
    if not seed.strip():
        st.error("시드를 입력하세요.")
        st.stop()

    # 세션에 입력값 저장(보고서 인터랙션 중에도 유지)
    st.session_state["seed"] = seed
    st.session_state["n_depth"] = n_depth

    seed_slug = slugify_seed(seed)
    base_dir = Path(BASE_RUN_DIR) / seed_slug
    seed_item_dir = base_dir / "seed_item"  # ✅ 확장/필터는 여기
    xtools_dir = base_dir / "xtools_item" / seed_slug
    ensure_dir(base_dir)
    ensure_dir(seed_item_dir)
    ensure_dir(xtools_dir)

    # 파일 경로들 (세션에 보관하여 rerun 시 재사용)
    paths = {
        "true": str(base_dir / "true_title.xlsx"),
        "expand": str(seed_item_dir / f"{n_depth}차시 확장 최종_결과.xlsx"),
        "filter": str(seed_item_dir / f"{seed_slug}_filtering_network.xlsx"),

        "xtools_dir": str(xtools_dir),
        "collect_flag": str(xtools_dir / ".collect_done"),

        # 아래 4개는 기본 예상 경로(통합 함수가 다른 이름으로 만들 수 있으니 'outs'로 대체 가능)
        "edit": str(xtools_dir / f"{seed_slug}_all_edit.xlsx"),
        "info": str(xtools_dir / f"{seed_slug}_all_info.xlsx"),
        "link": str(xtools_dir / f"{seed_slug}_all_link.xlsx"),
        "pageviews": str(xtools_dir / f"{seed_slug}_all_pageviews.xlsx"),

        "pagerank": str(base_dir / f"{seed_slug}_pagerank.xlsx"),
        "stats": str(base_dir / f"{seed_slug}_statistics.xlsx"),
    }
    ensure_dir(Path(paths["xtools_dir"]))

    # 1 True title
    with st.spinner("시드 확인 중…"):
        df_true = resolve_true_title(seed, paths["true"])
    seed_true = df_true.loc[0, "True_title"]
    seed_url = df_true.loc[0, "URL"] if "URL" in df_true.columns else ""
    seed_summary = df_true.loc[0, "Contents"] if "Contents" in df_true.columns else ""

    # 2 확장
    if not exists_nonempty(paths["expand"]) or not use_existing:
        paths["expand"] = expand_network(seed_true, paths["expand"], n_depth)
    df_expand = pd.read_excel(paths["expand"]) if exists_nonempty(paths["expand"]) else pd.DataFrame()
    st.subheader("📌 확장 결과 미리보기")
    st.write(f"총 {len(df_expand)}개의 링크")
    st.dataframe(df_expand.head(20))

    # 3 필터
    if not exists_nonempty(paths["filter"]) or not use_existing:
        paths["filter"] = filter_network(paths["expand"], seed_true, paths["filter"])
    df_filter = pd.read_excel(paths["filter"]) if exists_nonempty(paths["filter"]) else pd.DataFrame()
    st.subheader("📌 필터링 결과 미리보기")
    st.write(f"필터링 후 {len(df_filter)}개 남음")
    st.dataframe(df_filter.head(20))

    # 4 수집
    if not exists_nonempty(paths["collect_flag"]) or not use_existing:
        xtools_collect(seed_true, paths["filter"], paths["xtools_dir"], paths["collect_flag"])

    # 5 통합
    outs = xtools_integrate(seed_true, paths["xtools_dir"])
    # 세션에 실제 생성 경로 저장(보고서/추세그래프에서 사용)
    st.session_state["xtools_outs"] = outs

    # 6 PageRank — 없으면 계산해서 만들고, 있으면 읽기
    def _need_recompute(inp: str, out: str) -> bool:
        if not exists_nonempty(out, min_bytes=1):
            return True
        try:
            return os.path.getmtime(inp) > os.path.getmtime(out)  # 필터가 더 새로우면 재계산
        except Exception:
            return False

    if _need_recompute(paths["filter"], paths["pagerank"]) or not use_existing:
        pr_df = compute_pagerank(paths["filter"], paths["pagerank"])
    else:
        pr_df = pd.read_excel(paths["pagerank"]) if exists_nonempty(paths["pagerank"]) else pd.DataFrame()

    st.subheader("📌 PageRank Top 10")
    if pr_df.empty:
        st.info("PageRank 결과가 비어 있습니다. 네트워크가 너무 작거나 필터링이 과했을 수 있어요.")
    else:
        st.bar_chart(pr_df.set_index("title").head(10))

    # 7 통계
    if not exists_nonempty(paths["stats"]) or not use_existing:
        compute_statistics(
            edit_xlsx=outs.get("edit", paths["edit"]),
            pageviews_xlsx=outs.get("pageviews", paths["pageviews"]),
            link_xlsx=outs.get("link", paths["link"]),
            info_xlsx=outs.get("info", paths["info"]),
            pr_df=pr_df,
            out_xlsx=paths["stats"],
            ref_year=None,
        )
    _ = _read_excel(paths["stats"])  # warm cache

    # 세션에 산출 경로/요약 보관 (UI 조작 시 재사용)
    st.session_state["paths"] = paths
    st.session_state["seed_true"] = seed_true
    st.session_state["seed_url"] = seed_url
    st.session_state["seed_summary"] = seed_summary

# --- 보고서(세션에 결과가 있으면 표시) ---
if "paths" in st.session_state and exists_nonempty(st.session_state["paths"]["stats"]):
    paths = st.session_state["paths"]
    seed_true = st.session_state.get("seed_true", "")
    seed_url = st.session_state.get("seed_url", "")
    seed_summary = st.session_state.get("seed_summary", "")
    outs = st.session_state.get("xtools_outs", {})

    pr_df = pd.read_excel(paths["pagerank"]) if exists_nonempty(paths["pagerank"]) else pd.DataFrame()
    stats_df = pd.read_excel(paths["stats"]) if exists_nonempty(paths["stats"]) else pd.DataFrame()

    st.header("📄 보고서")

    # 1) 진입 아이템 / 2) 설명
    col_l, col_r = st.columns([1.1, 1.0], vertical_alignment="top")
    with col_l:
        st.subheader("1) 진입 아이템")
        st.markdown(f"**{seed_true}**")
        if seed_url:
            st.markdown(f"[Wikipedia]({seed_url})")
    with col_r:
        st.subheader("2) 아이템 설명")
        preview = (seed_summary or "").strip()
        if len(preview) > 320:
            st.write(preview[:320] + "…")
            with st.expander("더보기"):
                st.write(preview)
        else:
            st.write(preview if preview else "설명을 불러오지 못했습니다.")

    # 3) 전체 네트워크
    st.subheader("3) 전체 네트워크")
    st.caption("노드 크기 = PageRank (상위 일부 라벨)")

    view_mode = st.radio("네트워크 보기", ["정적(Matplotlib)", "인터랙티브(PyVis)"], horizontal=True)

    if view_mode == "정적(Matplotlib)":
        draw_network(paths["filter"], pr_df, max_nodes=120, label_top=15)
    else:
        html_path = render_pyvis_network(paths["filter"], pr_df, top_n=150, degree_min=1)
        if html_path:
            with open(html_path, "r", encoding="utf-8") as f:
                components.html(f.read(), height=650, scrolling=True)
        else:
            st.info("PyVis 그래프를 생성하지 못했습니다.")

    # 4) 유망성 지표 비교 (지표별 상위 아이템, 1~5점)
    st.subheader("4) 유망성 지표 비교 (지표별 상위 아이템, 1~5점)")
    INDICATORS = ["공급부상성", "수요부상성", "확산성", "기술집약도"]

    # 1~5 점수 컬럼 생성
    stats_scored = score_1to5(stats_df, INDICATORS) if not stats_df.empty else stats_df

    # UI 기본값을 먼저 세션에 세팅
    if "order_col" not in st.session_state:
        st.session_state["order_col"] = "기술집약도"
    if "top_n" not in st.session_state:
        st.session_state["top_n"] = 10

    # 선택 UI
    col_sel, col_n = st.columns([1.3, 1])
    with col_sel:
        order_col = st.selectbox(
            "정렬 기준 지표",
            INDICATORS,
            index=INDICATORS.index(st.session_state["order_col"]),
            key="order_col",
        )
    with col_n:
        top_n = st.slider(
            "상위 아이템 수",
            min_value=5, max_value=20, value=st.session_state["top_n"], step=1, key="top_n"
        )

    # 선택값으로 상위 N 추출
    ranked = stats_scored[["title"] + INDICATORS + [c + "_score" for c in INDICATORS]].copy() if not stats_df.empty else pd.DataFrame()
    if not ranked.empty:
        ranked = ranked.sort_values(order_col, ascending=False)
        top_items = ranked["title"].head(top_n).tolist()

        # --- 지표 선택 UI 추가 ---
        with st.expander("표시할 지표 선택", expanded=True):
            selected_indicators = st.multiselect(
                "표시할 지표를 선택하세요",
                INDICATORS,
                default=INDICATORS  # 기본은 모두 켜짐
            )

        if not selected_indicators:
            st.warning("적어도 하나 이상의 지표를 선택하세요.")
        else:
            radar_multi_by_items(
                stats_scored,
                top_items,
                selected_indicators,
                title=f"{order_col} 상위 {len(top_items)}개 아이템 — 선택된 지표만 (1~5점)",
                prefer_scores=True,
            )

        st.dataframe(ranked.head(top_n), use_container_width=True, height=420)
    else:
        st.info("지표를 계산할 데이터가 부족합니다.")

    # 5) 시드의 연도별 편집/조회 추세
    edit_path_for_plot = outs.get("edit", paths.get("edit", ""))
    view_path_for_plot = outs.get("pageviews", paths.get("pageviews", ""))
    if edit_path_for_plot or view_path_for_plot:
        plot_yearly_trends(seed_true, edit_path_for_plot, view_path_for_plot)

    # 다운로드
    st.markdown("---")
    st.markdown("### 📦 결과 다운로드")
    colA, colB = st.columns(2)
    with colA:
        if exists_nonempty(paths["stats"]):
            st.download_button(
                "통계 엑셀 다운로드",
                data=open(paths["stats"], "rb").read(),
                file_name=os.path.basename(paths["stats"]),
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
    with colB:
        if exists_nonempty(paths["pagerank"]):
            st.download_button(
                "PageRank 엑셀 다운로드",
                data=open(paths["pagerank"], "rb").read(),
                file_name=os.path.basename(paths["pagerank"]),
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
else:
    st.info("시드를 입력하고 ‘전체 실행’을 눌러 분석을 시작하세요.")
