# app.py
# -*- coding: utf-8 -*-
"""
Streamlit UI — 인터랙티브 대시보드.
모든 데이터 처리는 pipeline.py에 위임, UI/시각화만 담당.
"""

from __future__ import annotations
import os
import time
import threading
from pathlib import Path

import pandas as pd
import numpy as np
import networkx as nx
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import pipeline

# =====================
# Page Config & Theme
# =====================
st.set_page_config(
    page_title="유망아이템 분석도구",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

INDICATORS = ["공급부상성", "수요부상성", "확산성", "기술집약도"]

STEP_LABELS = {
    1: "시드 확인",
    2: "네트워크 확장 (위키 크롤링)",
    3: "네트워크 필터링",
    4: "XTools 수집 (편집/조회/링크)",
    5: "XTools 통합",
    6: "PageRank 계산",
    7: "통계 집계",
}

# Plotly 공통 테마
PLOTLY_THEME = dict(
    plot_bgcolor="rgba(248,249,250,1)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(size=12),
)
COLOR_PRIMARY = "#667eea"
COLOR_SECONDARY = "#764ba2"
COLOR_ACCENT = "#f093fb"

st.markdown("""
<style>
/* === Global === */
.block-container {
    max-width: 1200px;
    padding: 1rem 2rem 2rem 2rem;
}

/* === KPI Cards === */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 12px;
    padding: 16px 20px;
    color: white;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
}
div[data-testid="stMetric"] label {
    color: rgba(255,255,255,0.85) !important;
    font-size: 0.85rem !important;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: white !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
}

/* === Tab Styling === */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: #f0f2f6;
    border-radius: 10px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 8px 20px;
    font-weight: 600;
}
.stTabs [aria-selected="true"] {
    background-color: white;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}

/* === Header gradient === */
.gradient-header {
    background: linear-gradient(90deg, #667eea, #764ba2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 2.2rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
    display: inline-block;
    line-height: 1.4;
    padding-bottom: 0.1rem;
}
.sub-header {
    color: #555;
    font-size: 1rem;
    margin-bottom: 1.5rem;
}

/* === Card container === */
.info-card {
    background: white;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    border: 1px solid #e8eaed;
    margin-bottom: 1rem;
    line-height: 1.6;
}
.info-card h4 { margin: 0 0 0.3rem 0; color: #333; }
.info-card a { color: #667eea; text-decoration: none; }
.info-card p { margin: 0.2rem 0; color: #666; font-size: 0.92rem; }

/* === Download buttons === */
.stDownloadButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}
.stDownloadButton > button:hover {
    opacity: 0.9;
    box-shadow: 0 4px 12px rgba(102,126,234,0.4);
}

/* === Sidebar === */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
}
section[data-testid="stSidebar"] * {
    color: rgba(255,255,255,0.9) !important;
}
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] textarea,
section[data-testid="stSidebar"] [data-baseweb="input"] input,
section[data-testid="stSidebar"] [data-baseweb="textarea"] textarea {
    color: #1a1a2e !important;
    background-color: rgba(255,255,255,0.95) !important;
    border-color: rgba(255,255,255,0.4) !important;
}
section[data-testid="stSidebar"] input::placeholder,
section[data-testid="stSidebar"] textarea::placeholder {
    color: rgba(255,255,255,0.4) !important;
}
section[data-testid="stSidebar"] .stMarkdown hr {
    border-color: rgba(255,255,255,0.15);
}
section[data-testid="stSidebar"] button[kind="primary"] {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    border: none !important;
}
section[data-testid="stSidebar"] [data-testid="stNumberInputStepUp"],
section[data-testid="stSidebar"] [data-testid="stNumberInputStepDown"] {
    color: #fff !important;
    border-color: rgba(255,255,255,0.25) !important;
}
</style>
""", unsafe_allow_html=True)


# =====================
# 캐시 헬퍼
# =====================

@st.cache_data(show_spinner=False)
def _load_excel(path: str) -> pd.DataFrame:
    return pd.read_excel(path) if pipeline.exists_nonempty(path, 1) else pd.DataFrame()


# =====================
# KPI 요약 카드
# =====================

def render_kpi_row(pr_df: pd.DataFrame, stats_df: pd.DataFrame, edges_path: str):
    """대시보드 상단 KPI 요약 카드."""
    edges = pipeline.load_edges(edges_path) if pipeline.exists_nonempty(edges_path, 1) else pd.DataFrame()

    total_nodes = len(set(edges["From_title"].tolist() + edges["To_seealso"].tolist())) if not edges.empty else 0
    total_edges = len(edges)
    top_item = pr_df.iloc[0]["title"] if not pr_df.empty else "N/A"
    avg_pr = float(pr_df["pagerank"].mean()) if not pr_df.empty else 0.0
    avg_tech = float(stats_df["기술집약도"].mean()) if "기술집약도" in stats_df.columns else 0.0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("총 노드", f"{total_nodes:,}")
    c2.metric("총 엣지", f"{total_edges:,}")
    c3.metric("Top 아이템", top_item[:18])
    c4.metric("평균 PageRank", f"{avg_pr:.5f}")
    c5.metric("평균 기술집약도", f"{avg_tech:.5f}")


# =====================
# Plotly 시각화 함수
# =====================

def plotly_network(filtered_xlsx: str, pr_df: pd.DataFrame, max_nodes: int = 120, label_top: int = 15):
    """인터랙티브 네트워크 (Plotly scatter + edges)."""
    edges = pipeline.load_edges(filtered_xlsx)
    if edges.empty:
        st.info("네트워크가 비어 있습니다.")
        return

    H = pipeline._subgraph_for_viz(edges, pr_df, top_n=max_nodes, degree_min=1)
    if H.number_of_nodes() == 0:
        st.info("네트워크가 비어 있습니다.")
        return

    pos = nx.spring_layout(H, k=0.35, seed=42, iterations=50)
    pr_map = pr_df.set_index("title")["pagerank"] if not pr_df.empty else pd.Series(dtype=float)
    comm = pipeline._compute_communities(H)
    pr_max = float(pr_map.max()) if not pr_map.empty and pr_map.max() > 0 else 1.0

    palette = px.colors.qualitative.Set2
    n_comms = (max(comm.values()) + 1) if comm else 1

    # 엣지 (단일 trace, None 구분자)
    edge_x, edge_y = [], []
    for u, v in H.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=0.5, color="rgba(150,150,150,0.3)"),
        hoverinfo="none", showlegend=False,
    )

    # 노드 (커뮤니티별 trace)
    top_set = set(pr_df.head(label_top)["title"].tolist()) if not pr_df.empty else set()
    node_traces = []

    for c_id in range(n_comms):
        nodes = [n for n in H.nodes if comm.get(n, 0) == c_id]
        if not nodes:
            continue

        node_x = [pos[n][0] for n in nodes]
        node_y = [pos[n][1] for n in nodes]
        sizes = [8 + 40 * float(pr_map.get(n, 0.0)) / pr_max for n in nodes]
        hover = [
            f"<b>{n}</b><br>PageRank: {pr_map.get(n, 0.0):.6f}<br>Degree: {H.degree(n)}"
            for n in nodes
        ]
        labels = [n if n in top_set else "" for n in nodes]

        node_traces.append(go.Scatter(
            x=node_x, y=node_y, mode="markers+text",
            marker=dict(size=sizes, color=palette[c_id % len(palette)],
                        line=dict(width=0.5, color="white"), opacity=0.85),
            text=labels, textposition="top center", textfont=dict(size=9),
            hovertext=hover, hoverinfo="text", name=f"Community {c_id}",
        ))

    fig = go.Figure(data=[edge_trace] + node_traces)
    fig.update_layout(
        **PLOTLY_THEME,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=0, r=0, t=30, b=0),
        height=550, hovermode="closest",
    )
    st.plotly_chart(fig, width="stretch")


def plotly_bubble_scatter(stats_df: pd.DataFrame, top_n: int = 30):
    """버블 스캐터: X=공급부상성, Y=수요부상성, 크기=기술집약도, 색상=확산성."""
    if stats_df.empty or len(stats_df) < 2:
        st.info("버블 차트를 위한 데이터가 부족합니다.")
        return

    df = stats_df.nlargest(top_n, "기술집약도").copy()
    tech_min, tech_max = df["기술집약도"].min(), df["기술집약도"].max()
    if tech_max > tech_min:
        df["_size"] = 10 + 50 * (df["기술집약도"] - tech_min) / (tech_max - tech_min)
    else:
        df["_size"] = 30

    fig = px.scatter(
        df, x="공급부상성", y="수요부상성", size="_size",
        color="확산성", hover_name="title",
        hover_data={"기술집약도": ":.6f", "확산성": ":.2f", "_size": False},
        color_continuous_scale="Viridis", size_max=55,
    )
    fig.update_layout(
        **PLOTLY_THEME,
        title="유망성 지표 버블 차트",
        xaxis_title="공급부상성 (Supply Emergence)",
        yaxis_title="수요부상성 (Demand Emergence)",
        coloraxis_colorbar_title="확산성",
        height=480,
    )
    fig.update_traces(marker=dict(line=dict(width=1, color="white")))
    st.plotly_chart(fig, width="stretch")


def plotly_heatmap(wide_df: pd.DataFrame, title: str = "연도별 활동 히트맵", top_n: int = 25):
    """히트맵: articles × years 활동 강도."""
    if wide_df.empty or "title" not in wide_df.columns:
        st.info("히트맵 데이터가 없습니다.")
        return

    year_cols = sorted([c for c in wide_df.columns if str(c).isdigit() and 2000 <= int(str(c)) <= 2100])
    if not year_cols:
        st.info("연도 컬럼이 없습니다.")
        return

    df = wide_df.copy()
    df["_total"] = df[year_cols].sum(axis=1)
    df = df.nlargest(top_n, "_total")

    fig = go.Figure(data=go.Heatmap(
        z=df[year_cols].fillna(0).values,
        x=[str(y) for y in year_cols],
        y=df["title"].tolist(),
        colorscale="YlOrRd",
        hovertemplate="<b>%{y}</b><br>연도: %{x}<br>값: %{z:,.0f}<extra></extra>",
        colorbar=dict(title="활동량"),
    ))
    fig.update_layout(
        **PLOTLY_THEME,
        title=title,
        xaxis_title="연도",
        yaxis=dict(autorange="reversed"),
        height=max(400, top_n * 24),
        margin=dict(l=200),
    )
    st.plotly_chart(fig, width="stretch")


def plotly_treemap(pr_df: pd.DataFrame, stats_df: pd.DataFrame, top_n: int = 30):
    """트리맵: 면적 = PageRank, 색상 = 기술집약도."""
    if pr_df.empty:
        st.info("PageRank 데이터가 없습니다.")
        return

    df = pr_df.head(top_n).copy()
    if not stats_df.empty and "기술집약도" in stats_df.columns:
        df = df.merge(stats_df[["title", "기술집약도"]], on="title", how="left")
        df["기술집약도"] = df["기술집약도"].fillna(0)
    else:
        df["기술집약도"] = 0

    fig = px.treemap(
        df, path=["title"], values="pagerank",
        color="기술집약도", color_continuous_scale="Blues",
        title=f"PageRank 상위 {top_n}개 (면적 = PageRank, 색상 = 기술집약도)",
        hover_data={"pagerank": ":.6f", "기술집약도": ":.6f"},
    )
    fig.update_layout(height=480, margin=dict(t=50, l=10, r=10, b=10))
    fig.update_traces(textinfo="label+percent root", textfont=dict(size=11))
    st.plotly_chart(fig, width="stretch")


def plotly_time_series(edit_path: str, view_path: str, seed_title: str, key: str = ""):
    """인터랙티브 이중축 시계열 + range slider."""
    try:
        edit_wide = _load_excel(edit_path)
        view_wide = _load_excel(view_path)
    except Exception as e:
        st.warning(f"시계열 데이터 로드 오류: {e}")
        return

    s_ed = pipeline.extract_year_series(edit_wide, seed_title)
    s_vw = pipeline.extract_year_series(view_wide, seed_title)

    if s_ed.empty and s_vw.empty:
        st.info(f"'{seed_title}' 연도별 데이터가 없습니다.")
        return

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    if not s_ed.empty:
        fig.add_trace(go.Scatter(
            x=s_ed.index.tolist(), y=s_ed.values.tolist(),
            name="편집수", mode="lines+markers",
            line=dict(color=COLOR_PRIMARY, width=2.5), marker=dict(size=6),
            fill="tozeroy", fillcolor="rgba(102,126,234,0.08)",
        ), secondary_y=False)

    if not s_vw.empty:
        fig.add_trace(go.Scatter(
            x=s_vw.index.tolist(), y=s_vw.values.tolist(),
            name="조회수", mode="lines+markers",
            line=dict(color=COLOR_ACCENT, width=2.5), marker=dict(size=6),
            fill="tozeroy", fillcolor="rgba(240,147,251,0.08)",
        ), secondary_y=True)

    fig.update_layout(
        **PLOTLY_THEME,
        title=f"'{seed_title}' 연도별 활동 추세",
        xaxis=dict(title="연도", rangeslider=dict(visible=True), type="linear"),
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="편집수", secondary_y=False, gridcolor="rgba(0,0,0,0.05)")
    fig.update_yaxes(title_text="조회수", secondary_y=True, gridcolor="rgba(0,0,0,0.05)")
    chart_key = f"ts_{key or seed_title}"
    st.plotly_chart(fig, width="stretch", key=chart_key)


def plotly_radar(
    df_scores: pd.DataFrame,
    items: list[str],
    indicators: list[str],
    title: str = "유망성 레이더 차트",
):
    """인터랙티브 레이더 차트 (Plotly Scatterpolar)."""
    if len(items) < 3:
        st.info("레이더 차트는 최소 3개 이상의 아이템이 필요합니다.")
        return

    data = df_scores[df_scores["title"].isin(items)].set_index("title").reindex(items)
    palette = px.colors.qualitative.Set2

    fig = go.Figure()
    for i, indicator in enumerate(indicators):
        sc = f"{indicator}_score"
        col = sc if sc in data.columns else indicator
        vals = pd.to_numeric(data[col], errors="coerce").fillna(
            1.0 if col.endswith("_score") else 0.0
        ).tolist()
        vals += vals[:1]
        theta = items + [items[0]]

        fig.add_trace(go.Scatterpolar(
            r=vals, theta=theta, name=indicator,
            fill="toself",
            line=dict(color=palette[i % len(palette)], width=2),
            opacity=0.8,
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0.5, 5.5], tickvals=[1, 2, 3, 4, 5])),
        title=title, showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        height=550, **PLOTLY_THEME,
    )
    st.plotly_chart(fig, width="stretch")


def plotly_bar_top(pr_df: pd.DataFrame, top_n: int = 10, title: str = "PageRank Top 10"):
    """가로 막대 차트: PageRank 상위."""
    if pr_df.empty:
        st.info("PageRank 결과가 비어 있습니다.")
        return

    df = pr_df.head(top_n).iloc[::-1]  # 역순 (상위가 위로)
    fig = px.bar(
        df, x="pagerank", y="title", orientation="h",
        color="pagerank", color_continuous_scale="Purples",
        title=title,
    )
    fig.update_layout(
        **PLOTLY_THEME, height=max(300, top_n * 35),
        showlegend=False, coloraxis_showscale=False,
        xaxis_title="PageRank", yaxis_title="",
    )
    st.plotly_chart(fig, width="stretch")


# =====================
# 보고서 공통 위젯
# =====================

def _render_indicator_section(stats_df: pd.DataFrame, key_prefix: str = ""):
    """유망성 지표 비교 UI + Plotly 레이더 차트."""
    stats_scored = pipeline.score_1to5(stats_df, INDICATORS)

    order_key = f"{key_prefix}order_col"
    topn_key = f"{key_prefix}top_n"
    sel_key = f"{key_prefix}selected_indicators"

    if order_key not in st.session_state:
        st.session_state[order_key] = "기술집약도"
    if topn_key not in st.session_state:
        st.session_state[topn_key] = 10

    col_sel, col_n = st.columns([1.3, 1])
    with col_sel:
        order_col = st.selectbox(
            "정렬 기준 지표", INDICATORS,
            index=INDICATORS.index(st.session_state[order_key]),
            key=order_key,
        )
    with col_n:
        top_n = st.slider(
            "상위 아이템 수", min_value=5, max_value=20,
            value=st.session_state[topn_key], step=1, key=topn_key,
        )

    ranked = stats_scored[["title"] + INDICATORS + [c + "_score" for c in INDICATORS]].copy()
    ranked = ranked.sort_values(order_col, ascending=False)
    top_items = ranked["title"].head(top_n).tolist()

    with st.expander("표시할 지표 선택", expanded=True):
        selected_indicators = st.multiselect(
            "표시할 지표를 선택하세요", INDICATORS,
            default=INDICATORS, key=sel_key,
        )

    if not selected_indicators:
        st.warning("적어도 하나 이상의 지표를 선택하세요.")
    else:
        plotly_radar(
            stats_scored, top_items, selected_indicators,
            title=f"{order_col} 상위 {len(top_items)}개 아이템 (1~5점)",
        )

    st.dataframe(ranked.head(top_n), width="stretch", height=420)


# =====================
# 보고서 렌더링
# =====================

def _render_single_report():
    """단일 시드 보고서 (탭 레이아웃)."""
    paths = st.session_state["paths"]
    outs = st.session_state.get("outs", {})
    seed_true = st.session_state.get("seed_true", "")
    seed_url = st.session_state.get("seed_url", "")
    seed_summary = st.session_state.get("seed_summary", "")

    pr_df = _load_excel(paths["pagerank"])
    stats_df = _load_excel(paths["stats"])

    # 헤더 + 시드 카드
    st.markdown(f'<div class="gradient-header">📄 {seed_true}</div>', unsafe_allow_html=True)

    preview = (seed_summary or "").strip()
    summary_display = (preview[:300] + "...") if len(preview) > 300 else preview
    st.markdown(f"""
    <div class="info-card">
        <h4>{seed_true}</h4>
        <p><a href="{seed_url}" target="_blank">🔗 Wikipedia 원문 보기</a></p>
        <p>{summary_display or '설명 없음'}</p>
    </div>
    """, unsafe_allow_html=True)

    # KPI 요약
    render_kpi_row(pr_df, stats_df, paths["filter"])

    # 탭 보고서
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📊 Overview", "🕸 Network", "📈 Indicators", "📉 Trends", "📋 Data"]
    )

    with tab1:
        col_a, col_b = st.columns(2)
        with col_a:
            plotly_treemap(pr_df, stats_df, top_n=20)
        with col_b:
            plotly_bubble_scatter(stats_df, top_n=25)
        plotly_bar_top(pr_df, top_n=10)

    with tab2:
        st.caption("노드 크기 = PageRank, 색상 = Community 그룹. 마우스 드래그로 확대, 호버로 상세 정보 확인")
        c1, c2 = st.columns(2)
        with c1:
            max_n = st.slider("최대 노드 수", 30, 200, 120, key="net_max_nodes")
        with c2:
            label_n = st.slider("라벨 표시 (상위 N)", 5, 30, 15, key="net_label_top")
        plotly_network(paths["filter"], pr_df, max_nodes=max_n, label_top=label_n)

    with tab3:
        _render_indicator_section(stats_df)

    with tab4:
        edit_path = outs.get("edit", paths.get("edit", ""))
        view_path = outs.get("pageviews", paths.get("pageviews", ""))
        if pipeline.exists_nonempty(edit_path, 1) and pipeline.exists_nonempty(view_path, 1):
            plotly_time_series(edit_path, view_path, seed_true)
            st.markdown("---")
            hm_mode = st.radio("히트맵 데이터", ["편집수", "조회수"], horizontal=True, key="hm_mode")
            if hm_mode == "편집수":
                plotly_heatmap(_load_excel(edit_path), title="연도별 편집수 히트맵", top_n=25)
            else:
                plotly_heatmap(_load_excel(view_path), title="연도별 조회수 히트맵", top_n=25)
        else:
            st.info("트렌드 데이터가 없습니다.")

    with tab5:
        st.subheader("통계 데이터")
        st.dataframe(stats_df, width="stretch", height=500)
        st.subheader("PageRank")
        st.dataframe(pr_df, width="stretch", height=400)
        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "📥 통계 다운로드",
                data=open(paths["stats"], "rb").read(),
                file_name=os.path.basename(paths["stats"]),
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        with c2:
            st.download_button(
                "📥 PageRank 다운로드",
                data=open(paths["pagerank"], "rb").read(),
                file_name=os.path.basename(paths["pagerank"]),
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )


def _render_multi_report():
    """멀티 시드 통합 보고서 (탭 레이아웃)."""
    multi_paths = st.session_state["multi_paths"]
    per_seed_results = st.session_state["per_seed_results"]

    pr_df = _load_excel(multi_paths["merged_pagerank"])
    stats_df = _load_excel(multi_paths["merged_stats"])

    # 헤더
    st.markdown('<div class="gradient-header">📄 멀티 시드 통합 보고서</div>', unsafe_allow_html=True)

    # 시드 목록 카드
    seed_info_rows = []
    for r in per_seed_results:
        _, _, s_true, s_url, s_summary = r
        preview = (s_summary or "").strip()[:200]
        seed_info_rows.append({"시드": s_true, "URL": s_url or "", "요약": preview})
    st.dataframe(pd.DataFrame(seed_info_rows), width="stretch")

    # KPI 요약
    render_kpi_row(pr_df, stats_df, multi_paths["merged_edges"])

    # 탭 보고서
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📊 Overview", "🕸 Network", "📈 Indicators", "📉 Trends", "📋 Data"]
    )

    with tab1:
        col_a, col_b = st.columns(2)
        with col_a:
            plotly_treemap(pr_df, stats_df, top_n=30)
        with col_b:
            plotly_bubble_scatter(stats_df, top_n=30)
        plotly_bar_top(pr_df, top_n=10, title="통합 PageRank Top 10")

    with tab2:
        st.caption("노드 크기 = PageRank, 색상 = Community 그룹")
        c1, c2 = st.columns(2)
        with c1:
            max_n = st.slider("최대 노드 수", 30, 200, 120, key="m_net_max_nodes")
        with c2:
            label_n = st.slider("라벨 표시 (상위 N)", 5, 30, 15, key="m_net_label_top")
        plotly_network(multi_paths["merged_edges"], pr_df, max_nodes=max_n, label_top=label_n)

    with tab3:
        _render_indicator_section(stats_df, key_prefix="m_")

    with tab4:
        st.subheader("시드별 연도 트렌드")
        for i, r in enumerate(per_seed_results):
            r_paths, r_outs, s_true, _, _ = r
            with st.expander(f"📌 {s_true}"):
                edit_path = r_outs.get("edit", r_paths["edit"])
                view_path = r_outs.get("pageviews", r_paths["pageviews"])
                if pipeline.exists_nonempty(edit_path, 1) and pipeline.exists_nonempty(view_path, 1):
                    plotly_time_series(edit_path, view_path, s_true, key=f"multi_{i}_{s_true}")
                else:
                    st.info("트렌드 데이터가 없습니다.")

        # 통합 히트맵
        merged_outs = st.session_state.get("merged_outs", {})
        st.markdown("---")
        hm_mode = st.radio("통합 히트맵 데이터", ["편집수", "조회수"], horizontal=True, key="m_hm_mode")
        if hm_mode == "편집수" and merged_outs.get("edit"):
            plotly_heatmap(_load_excel(merged_outs["edit"]), title="통합 편집수 히트맵", top_n=30)
        elif merged_outs.get("pageviews"):
            plotly_heatmap(_load_excel(merged_outs["pageviews"]), title="통합 조회수 히트맵", top_n=30)

    with tab5:
        st.subheader("통합 통계")
        st.dataframe(stats_df, width="stretch", height=500)
        st.subheader("통합 PageRank")
        st.dataframe(pr_df, width="stretch", height=400)
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.download_button(
                "📥 통합 통계",
                data=open(multi_paths["merged_stats"], "rb").read(),
                file_name="merged_statistics.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        with c2:
            st.download_button(
                "📥 통합 PageRank",
                data=open(multi_paths["merged_pagerank"], "rb").read(),
                file_name="merged_pagerank.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        with c3:
            st.download_button(
                "📥 통합 엣지",
                data=open(multi_paths["merged_edges"], "rb").read(),
                file_name="merged_edges.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        # --- 최종 교집합 파일 ---
        final = multi_paths.get("final", {})
        if final:
            st.markdown("---")
            st.subheader("최종 교집합 파일 (pagerank ∩ edit ∩ pageviews)")
            final_stats = final.get("stats", "")
            if final_stats and pipeline.exists_nonempty(final_stats):
                final_df = _load_excel(final_stats)
                st.caption(f"공통 아이템: {len(final_df)}개")
                st.dataframe(final_df, width="stretch", height=400)

            fc1, fc2, fc3 = st.columns(3)
            for col, label, key in [
                (fc1, "📥 최종 통계", "stats"),
                (fc2, "📥 최종 PageRank", "pagerank"),
                (fc3, "📥 최종 Edit", "edit"),
            ]:
                fp = final.get(key, "")
                if fp and pipeline.exists_nonempty(fp):
                    with col:
                        st.download_button(
                            label,
                            data=open(fp, "rb").read(),
                            file_name=os.path.basename(fp),
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key=f"dl_final_{key}",
                        )


# =====================
# 파이프라인 모니터링
# =====================

def _monitor_pipeline(mon, seed_label=""):
    """백그라운드 파이프라인 모니터링 루프 + 결과 처리.

    session_state['pipeline_mon']의 상태에 따라:
    - running → 진행률 UI 표시하며 완료 대기
    - done    → 결과를 session_state에 저장
    - error   → 에러 표시 후 중단
    """
    _STEP_ICONS = {1: "📋", 2: "🔄", 3: "✂️", 4: "📡", 5: "🔗", 6: "📊", 7: "📈"}

    progress_ph = st.empty()
    monitor_ph = st.empty()
    collect_ph = st.empty()

    if mon.get("status") == "running":
        while mon["status"] == "running":
            cur = mon["current_step"]

            if cur > 0 and cur <= 7:
                if cur == 4 and mon.get("collect_total", 0) > 0:
                    sub = mon["collect_idx"] / mon["collect_total"]
                    pct = (3 + sub) / 7
                else:
                    pct = max(0, cur - 1) / 7
                progress_ph.progress(min(pct, 0.99),
                                     text=f"Step {cur}/7 — {STEP_LABELS.get(cur, '')}")

            lines = []
            for s_num in range(1, 8):
                if s_num < cur:
                    icon = "✅"
                elif s_num == cur:
                    icon = "🔄"
                else:
                    icon = "⬜"
                s_icon = _STEP_ICONS.get(s_num, "")
                label = STEP_LABELS.get(s_num, f"단계 {s_num}")
                line = f"{s_icon} **Step {s_num}: {label}** {icon}"

                if s_num == 2 and mon.get("expand_iters"):
                    for it in mon["expand_iters"]:
                        line += f"\n  - {it['iteration']}차시: frontier {it['frontier']} → 수집 {it['collected']}건"
                    if mon.get("expand_done"):
                        d = mon["expand_done"]
                        line += f"\n  - **총 방문: {d.get('nodes', 0)}노드 | 총 엣지: {d.get('edges', 0)}건**"
                elif s_num == 3 and mon.get("filter_done"):
                    fd = mon["filter_done"]
                    line += (f"\n  > {fd.get('before', 0)}건 → {fd.get('after', 0)}건"
                             f" ({fd.get('removed_pct', 0)}% 제거)"
                             f" | 확정 문서: {fd.get('nodes', 0)}개")
                elif s_num == 4 and mon.get("collect_total", 0) > 0:
                    line += f" [{mon['collect_idx']}/{mon['collect_total']}]"
                elif s_num == 6 and mon.get("pagerank_top5"):
                    for rank, item in enumerate(mon["pagerank_top5"], 1):
                        line += f"\n  - Top {rank}: **{item['title']}** ({item['pagerank']:.6f})"
                elif s_num == 7 and mon.get("stats_done"):
                    sd = mon["stats_done"]
                    line += (f"\n  > 총 {sd.get('total_items', 0)}개 아이템"
                             f" | 평균 기술집약도: {sd.get('avg_tech', 0):.6f}")

                lines.append(line)

            monitor_ph.markdown("\n\n".join(lines))

            if cur == 4 and mon.get("collect_nodes"):
                rows = []
                for cn in mon["collect_nodes"]:
                    s = cn.get("status", "pending")
                    if s == "skipped":
                        st_txt, e, p, inf, lnk = "⏭", "⏭", "⏭", "⏭", "⏭"
                    elif s == "done":
                        st_txt = "✅"
                        e = "✅" if cn.get("edit") else "❌"
                        p = "✅" if cn.get("pageviews") else "❌"
                        inf = "✅" if cn.get("info") else "❌"
                        lnk = "✅" if cn.get("link") else "❌"
                    elif s == "running":
                        st_txt, e, p, inf, lnk = "🔄", "🔄", "🔄", "🔄", "🔄"
                    else:
                        st_txt, e, p, inf, lnk = "⬜", "⬜", "⬜", "⬜", "⬜"
                    rows.append({"문서명": cn["node"][:30], "edit": e, "pv": p,
                                 "info": inf, "link": lnk, "상태": st_txt})
                df_disp = pd.DataFrame(rows)
                if len(df_disp) > 12:
                    center = max(0, mon["collect_idx"] - 3)
                    df_disp = df_disp.iloc[center:center + 12]
                collect_ph.dataframe(df_disp, width="stretch", hide_index=True)
            else:
                collect_ph.empty()

            time.sleep(2)

    # --- 모니터링 종료 후 정리 ---
    collect_ph.empty()

    if mon.get("status") == "error":
        progress_ph.progress(1.0, text=f"'{seed_label}' 분석 실패")
        st.error(f"분석 실패: {mon.get('error', '알 수 없는 오류')}")
        st.session_state.pop("pipeline_mon", None)
        st.stop()

    if mon.get("status") == "done" and "result" in mon:
        progress_ph.progress(1.0, text=f"'{seed_label}' 분석 완료!")
        paths, outs, seed_true, seed_url, seed_summary = mon["result"]
        st.session_state.pop("pipeline_mon", None)
        st.session_state.update(
            paths=paths, outs=outs,
            seed_true=seed_true, seed_url=seed_url, seed_summary=seed_summary,
        )


# ============================
# 사이드바: 입력 컨트롤
# ============================

with st.sidebar:
    st.markdown("## 🧭 분석 설정")
    st.markdown("---")

    analysis_mode = st.radio("분석 모드", ["단일 시드", "멀티 시드"])

    if analysis_mode == "단일 시드":
        seed = st.text_input("시드 입력", placeholder="예: Nvidia",
                             value=st.session_state.get("seed", ""))
        seeds_list = [seed] if seed.strip() else []
    else:
        seeds_raw = st.text_area(
            "시드 목록 (줄바꿈으로 구분)",
            placeholder="예:\nElectric battery\nSolar cell\nWind power",
            height=120,
            value=st.session_state.get("seeds_raw", ""),
        )
        seeds_list = [s.strip() for s in seeds_raw.strip().split("\n") if s.strip()]
        if seeds_list:
            st.caption(f"입력된 시드: {len(seeds_list)}개")

    st.markdown("---")
    n_depth = st.number_input(
        "확장 차수 (n)", min_value=1, max_value=3,
        value=st.session_state.get("n_depth", 1), step=1,
    )
    use_existing = st.toggle("기존 산출물 재사용", value=True)

    st.markdown("---")
    run = st.button("🚀 전체 실행", width="stretch", type="primary")


# ============================
# 메인: 타이틀 + 실행 + 보고서
# ============================

# --- 실행 ---
if run:
    if not seeds_list:
        st.error("시드를 입력하세요.")
        st.stop()

    st.session_state["n_depth"] = n_depth
    st.session_state["analysis_mode"] = analysis_mode

    # ===== 단일 시드 =====
    if analysis_mode == "단일 시드":
        seed = seeds_list[0]
        st.session_state["seed"] = seed

        # --- 백그라운드 파이프라인 상태를 session_state에 저장 ---
        # (Streamlit 리렌더링되어도 유지됨)
        if "pipeline_mon" not in st.session_state:
            st.session_state["pipeline_mon"] = {}
        mon = st.session_state["pipeline_mon"]
        mon.update({
            "status": "running",  # running / done / error
            "current_step": 0,
            "error": None,
            "expand_iters": [],
            "expand_done": {},
            "filter_done": {},
            "collect_nodes": [],
            "collect_total": 0,
            "collect_idx": 0,
            "pagerank_top5": [],
            "pagerank_total": 0,
            "stats_done": {},
        })

        # --- 콜백 함수 (백그라운드 스레드에서 session_state 업데이트) ---
        def _on_progress(step, total, msg):
            mon["current_step"] = step

        def _on_detail(event, data):
            if event == "expand_iter":
                mon["expand_iters"].append(data)
            elif event == "expand_done":
                mon["expand_done"] = data
            elif event == "filter_done":
                mon["filter_done"] = data
            elif event == "collect_start":
                nodes_list = data.get("nodes", [])
                mon["collect_total"] = data.get("total", len(nodes_list))
                mon["collect_idx"] = 0
                mon["collect_nodes"] = [
                    {"node": n, "edit": False, "pageviews": False,
                     "info": False, "link": False, "status": "pending"}
                    for n in nodes_list
                ]
            elif event == "collect_node":
                idx = data.get("idx", 0)
                mon["collect_idx"] = idx + 1
                if idx < len(mon["collect_nodes"]):
                    cn = mon["collect_nodes"][idx]
                    cn["edit"] = data.get("edit", False)
                    cn["pageviews"] = data.get("pageviews", False)
                    cn["info"] = data.get("info", False)
                    cn["link"] = data.get("link", False)
                    cn["status"] = "skipped" if data.get("skipped") else "done"
                if idx + 1 < len(mon["collect_nodes"]):
                    mon["collect_nodes"][idx + 1]["status"] = "running"
            elif event == "collect_done":
                pass
            elif event == "pagerank_done":
                mon["pagerank_top5"] = data.get("top5", [])
                mon["pagerank_total"] = data.get("total", 0)
            elif event == "stats_done":
                mon["stats_done"] = data

        # --- 백그라운드 스레드에서 파이프라인 실행 ---
        def _run_pipeline():
            try:
                result = pipeline.run_analysis_pipeline(
                    seed, n_depth, use_existing,
                    on_progress=_on_progress, on_detail=_on_detail,
                )
                mon["result"] = result
                mon["status"] = "done"
                mon["current_step"] = 8
            except Exception as e:
                mon["error"] = str(e)
                mon["status"] = "error"

        pipeline_thread = threading.Thread(target=_run_pipeline, daemon=True)
        pipeline_thread.start()

        # --- 메인 스레드: 모니터링 루프 ---
        _monitor_pipeline(mon, seed)

    # ===== 멀티 시드 =====
    else:
        st.session_state["seeds_raw"] = seeds_raw
        st.session_state["seeds_list"] = seeds_list

        per_seed_results = []
        failed_seeds = []

        progress_bar = st.progress(0, text="시드별 분석 준비 중...")
        status = st.status("시드별 분석 진행 중...", expanded=True)
        multi_collect_ph = st.empty()

        for i, s in enumerate(seeds_list):
            if i > 0:
                import time; time.sleep(2)

            progress_bar.progress(
                i / len(seeds_list),
                text=f"[{i + 1}/{len(seeds_list)}] '{s}' 분석 중...",
            )
            status.update(label=f"[{i + 1}/{len(seeds_list)}] '{s}' 분석 중...", state="running")

            _multi_mon = {"collect_idx": 0, "collect_total": 0}

            def _on_progress(step, total, msg, _i=i, _s=s):
                label = STEP_LABELS.get(step, f"단계 {step}")
                pct = (_i + step / total) / len(seeds_list)
                progress_bar.progress(
                    min(pct, 0.99),
                    text=f"[{_i + 1}/{len(seeds_list)}] '{_s}' — {step}/{total} {label}",
                )
                status.write(f"  `{_s}` : {step}/{total} {label}")

            def _on_detail_multi(event, data, _s=s):
                if event == "collect_start":
                    _multi_mon["collect_total"] = data.get("total", 0)
                    _multi_mon["collect_idx"] = 0
                elif event == "collect_node":
                    _multi_mon["collect_idx"] = data.get("idx", 0) + 1
                    t = _multi_mon["collect_total"]
                    c = _multi_mon["collect_idx"]
                    if t > 0:
                        multi_collect_ph.progress(
                            c / t,
                            text=f"'{_s}' XTools 수집: {c}/{t} ({c * 100 // t}%)"
                        )
                elif event == "collect_done":
                    multi_collect_ph.empty()

            try:
                result = pipeline.run_analysis_pipeline(
                    s, n_depth, use_existing,
                    on_progress=_on_progress, on_detail=_on_detail_multi,
                )
                per_seed_results.append(result)
                status.write(f"  **{s}** — 완료")
            except Exception as e:
                failed_seeds.append((s, str(e)))
                status.write(f"  **{s}** — 실패: {e}")
                st.warning(f"'{s}' 실패: {e}")

        progress_bar.progress(1.0, text="시드별 분석 완료")
        n_ok, n_fail = len(per_seed_results), len(failed_seeds)

        if failed_seeds:
            status.update(label=f"시드별 분석 완료 (성공 {n_ok}, 실패 {n_fail})", state="error")
        else:
            status.update(label=f"시드별 분석 완료 ({n_ok}개 성공)", state="complete")

        if not per_seed_results:
            st.error("모든 시드가 실패했습니다.")
            st.stop()

        if failed_seeds:
            st.warning(f"{n_fail}개 시드 실패: {[f[0] for f in failed_seeds]}")

        # Phase 2-5: 병합
        run_hash = pipeline.multi_seed_hash(seeds_list, n_depth)
        multi_dir = Path(pipeline.BASE_RUN_DIR) / f"multi_{run_hash}"
        merged_xtools_dir = multi_dir / "merged_xtools"
        pipeline.ensure_dir(multi_dir)
        pipeline.ensure_dir(merged_xtools_dir)

        multi_paths = {
            "merged_edges": str(multi_dir / "merged_edges.xlsx"),
            "merged_pagerank": str(multi_dir / "merged_pagerank.xlsx"),
            "merged_stats": str(multi_dir / "merged_statistics.xlsx"),
            "merged_xtools_dir": str(merged_xtools_dir),
        }

        with st.spinner("필터링된 네트워크 병합 중..."):
            filter_files = [r[0]["filter"] for r in per_seed_results]
            seed_true_names = [r[2] for r in per_seed_results]
            pipeline.merge_filtered_edges(filter_files, multi_paths["merged_edges"],
                                          seed_nodes=seed_true_names)

        with st.spinner("XTools 데이터 병합 중..."):
            per_seed_outs = [r[1] for r in per_seed_results]
            merged_outs = pipeline.merge_xtools_data(per_seed_outs, str(merged_xtools_dir))

        with st.spinner("통합 PageRank 계산 중..."):
            pipeline.compute_pagerank(multi_paths["merged_edges"], multi_paths["merged_pagerank"],
                                     seed_nodes=seed_true_names)

        with st.spinner("통합 통계 산출 중..."):
            _pr = _load_excel(multi_paths["merged_pagerank"])
            pipeline.compute_statistics(
                edit_xlsx=merged_outs["edit"],
                pageviews_xlsx=merged_outs["pageviews"],
                link_xlsx=merged_outs["link"],
                info_xlsx=merged_outs["info"],
                pr_df=_pr,
                out_xlsx=multi_paths["merged_stats"],
                ref_year=None,
            )

        with st.spinner("최종 교집합 파일 생성 중..."):
            seed_true_names = [r[2] for r in per_seed_results]
            final_paths = pipeline.finalize_merged(
                stats_xlsx=multi_paths["merged_stats"],
                pagerank_xlsx=multi_paths["merged_pagerank"],
                merged_outs=merged_outs,
                out_dir=str(multi_dir / "final"),
                seed_titles=seed_true_names,
            )
            multi_paths["final"] = final_paths

        st.session_state["multi_paths"] = multi_paths
        st.session_state["per_seed_results"] = per_seed_results
        st.session_state["merged_outs"] = merged_outs

        # --- 시드 포함 검증 결과 표시 ---
        seed_check = final_paths.get("_seed_check", {})
        if seed_check:
            total = seed_check["total"]
            found = seed_check["found"]
            missing = seed_check.get("missing", [])
            if missing:
                st.warning(f"멀티 시드 분석 완료! (성공: {n_ok}개, 실패: {n_fail}개)\n\n"
                           f"최종 파일 시드 검증: {found}/{total}개 포함, "
                           f"누락: {', '.join(missing)}")
            else:
                st.success(f"멀티 시드 분석 완료! (성공: {n_ok}개, 실패: {n_fail}개)\n\n"
                           f"최종 파일 시드 검증: {found}/{total}개 전부 포함 ✓")
        else:
            st.success(f"멀티 시드 분석 완료! (성공: {n_ok}개, 실패: {n_fail}개)")


# --- 백그라운드 파이프라인 복구 (브라우저 재연결/새로고침 시) ---
if not run and "pipeline_mon" in st.session_state:
    _monitor_pipeline(
        st.session_state["pipeline_mon"],
        st.session_state.get("seed", ""),
    )

# --- 보고서 ---
_mode = st.session_state.get("analysis_mode", "단일 시드")

if _mode == "멀티 시드" and "multi_paths" in st.session_state:
    if pipeline.exists_nonempty(st.session_state["multi_paths"].get("merged_stats", "")):
        _render_multi_report()
    else:
        st.info("👈 사이드바에서 시드를 입력하고 실행하세요.")

elif "paths" in st.session_state and pipeline.exists_nonempty(st.session_state["paths"].get("stats", "")):
    _render_single_report()

else:
    st.markdown("""
    <div style="text-align:center; padding: 4rem 1rem 2rem 1rem;">
        <div class="gradient-header" style="font-size: 2.5rem; margin-bottom: 0.3rem;">🧭 유망아이템 분석도구</div>
        <div style="color: #888; font-size: 1.05rem; margin-bottom: 2.5rem;">
            Wikipedia 네트워크 기반 유망 기술/아이템 탐색 & 분석 대시보드
        </div>
        <div style="color: #aaa; font-size: 1rem;">
            👈 사이드바에서 시드를 입력하고 <b style="color:#667eea;">전체 실행</b>을 눌러 시작하세요
        </div>
    </div>
    """, unsafe_allow_html=True)
