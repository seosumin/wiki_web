# -*- coding: utf-8 -*-
"""
교육용 '유망아이템 발굴' 단계별 학습/시연 Streamlit 앱.

- 좌측: 사용자·분석 설정 (데모 프리셋 / 자유 실습).
- 본문: 7단계 카드. 각 단계마다 '이 단계에서 무슨 일이?' 설명 + [실행] 버튼 +
        결과 시각화(표/추세 라인차트/네트워크 그래프/부상도 맵/AI 보고서).
- 강사용: [전체 자동 실행] 한 번에 7단계 시연.

실행:
  PYTHONUTF8=1 ./venv/Scripts/streamlit.exe run edu/app.py
"""
from __future__ import annotations

import os
import sys

# `streamlit run edu/app.py`는 스크립트 폴더(edu/)만 sys.path에 넣어 `import edu`가 실패한다.
# 실행 위치와 무관하게 동작하도록 프로젝트 루트(edu/의 상위)를 경로에 추가한다.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from edu import apollo, crawl, pipeline, store

st.set_page_config(page_title="유망아이템 발굴 실습", page_icon="🔭", layout="wide")

# ------------------------------------------------------------------
# 세션 상태
# ------------------------------------------------------------------
ss = st.session_state
ss.setdefault("run_id", None)
ss.setdefault("cfg", {})


def _reset_run(run_id: str, cfg: dict) -> None:
    ss.run_id = run_id
    ss.cfg = cfg


def _steps_map(run_id: str) -> dict:
    return {s["step_no"]: s for s in store.get_steps(run_id)}


def _done(steps: dict, no: int) -> bool:
    return steps.get(no, {}).get("status") == "done"


@st.cache_data(show_spinner=False, ttl=1800)
def _cached_top100(category: str, indicator: str) -> list:
    """시드 선택 UI용 TOP100 (30분 캐시 — 사이드바에서 반복 호출 방지)."""
    return apollo.top100(category, indicator)


# ------------------------------------------------------------------
# 사이드바 — 설정
# ------------------------------------------------------------------
with st.sidebar:
    st.header("🔧 분석 설정")

    user_name = st.text_input("이름 / 조", value="guest")

    mode_label = st.radio("모드", ["🎬 데모 실습 (APOLLO)", "🕸 직접 수집 (위키 크롤링)"], index=0)
    kind = "crawl" if mode_label.startswith("🕸") else "apollo"

    category = indicator = None
    seed_rank = 1
    seed_text, n_depth = "", 1
    net_degree = 3

    if kind == "apollo":  # 데모 실습 (큐레이션된, 네트워크가 잘 잡히는 사례)
        preset_labels = [p["label"] for p in pipeline.DEMO_PRESETS]
        pi = st.selectbox("데모 사례", range(len(preset_labels)),
                          format_func=lambda i: preset_labels[i])
        preset = pipeline.DEMO_PRESETS[pi]
        category = preset["category"]
        indicator = preset["indicator"]
        seed_rank = preset["seed_rank"]
        st.caption(f"카테고리: **{category}** · 지표: **{apollo.INDICATORS[indicator]}** · 시드 순위: {seed_rank}")
        with st.expander("고급 옵션"):
            net_degree_ui = st.select_slider(
                "네트워크 확장 차수 (2단계)", options=[1, 2, 3], value=2,
                help="시드에서 몇 단계까지 연관망을 넓힐지. 1차수=직접 이웃, "
                     "2차수=이웃의 이웃까지(실제 네트워크 분석에 적합).")
            # 사용자 '차수'는 APOLLO degree-1 (degree=1은 빈 결과).
            net_degree = net_degree_ui + 1
    else:  # 🕸 직접 수집 (위키 크롤링)
        seed_text = st.text_input("수집할 아이템 (영문 위키 제목 권장)", value="Quantum computing")
        n_depth = 1  # 1차수 고정 (2차수 이상은 XTools 수집이 과도하게 느려 데모 부적합)
        st.caption("⚠️ 위키/XTools를 **1차수**로 직접 크롤링합니다. 아이템당 **수 분** 소요 (심화·시연용).")

    st.divider()
    mode = "crawl" if kind == "crawl" else "demo"
    if st.button("① 새 분석 시작", type="primary", width="stretch"):
        seed_query = seed_text if kind == "crawl" else f"{category}/{indicator}"
        rid = store.create_run(user_name, seed_query, mode=mode)
        _reset_run(rid, dict(kind=kind, category=category, indicator=indicator, seed_rank=seed_rank,
                             net_degree=net_degree, seed_text=seed_text, n_depth=n_depth,
                             user_name=user_name))
        st.rerun()

    run_all_clicked = False
    if kind == "apollo":
        run_all_clicked = st.button("⚡ 전체 자동 실행 (강사 시연)", width="stretch",
                                    disabled=ss.run_id is None)

    st.divider()
    st.caption(f"DB: `{store.DB_BACKEND}`  ·  APOLLO A4 / 위키 크롤링")
    with st.expander("이전 실행 기록"):
        for r in store.list_runs(limit=15):
            tag = {"demo": "🎬", "crawl": "🕸"}.get(r["mode"], "🧪")
            if st.button(f"{tag} {r['seed_query']} · {r['status']}", key=f"run_{r['run_id']}",
                         width="stretch"):
                _reset_run(r["run_id"], dict(
                    kind=("crawl" if r["mode"] == "crawl" else "apollo"),
                    category=r["category"], indicator="TECH_INTENSITY",
                    seed_rank=1, net_degree=3,
                    seed_text=r["seed_query"], n_depth=1, user_name=r["user_name"]))
                st.rerun()


# ------------------------------------------------------------------
# 헤더
# ------------------------------------------------------------------
st.title("🔭 유망아이템 발굴 — 단계별 실습")
st.markdown(
    "KISTI **APOLLO** 공식 데이터로 *시드 확정 → 연관 네트워크 확장 → 상세 수집 → "
    "종합 → AI 보고서*까지 단계별로 실행하며 배웁니다."
)

if ss.run_id is None:
    st.info("👈 왼쪽에서 설정을 고르고 **① 새 분석 시작**을 눌러 주세요.")
    st.stop()

run_id = ss.run_id
cfg = ss.cfg
steps = _steps_map(run_id)

# 진행 상황 트래커 (APOLLO 모드 전용)
if cfg.get("kind") != "crawl":
    cols = st.columns(pipeline.TOTAL_STEPS)
    for i, no in enumerate(range(1, pipeline.TOTAL_STEPS + 1)):
        stt = steps.get(no, {}).get("status", "pending")
        icon = {"done": "✅", "running": "⏳", "error": "❌"}.get(stt, "⚪")
        cols[i].markdown(f"<div style='text-align:center'>{icon}<br><small>{no}. {pipeline.STEP_NAMES[no]}</small></div>",
                         unsafe_allow_html=True)
    st.divider()


# ------------------------------------------------------------------
# 전체 자동 실행
# ------------------------------------------------------------------
def _run_all():
    bar = st.progress(0, text="시작…")

    def prog(s, t, m):
        bar.progress(int(s / t * 100), text=f"[{s}/{t}] {m}")

    pipeline.run_all(run_id, cfg["category"], cfg["indicator"], cfg["seed_rank"],
                     net_degree=cfg.get("net_degree", 3), progress=prog)
    bar.progress(100, text="완료")


if run_all_clicked:
    with st.spinner("전체 단계 자동 실행 중…"):
        _run_all()
    st.success("전체 실행 완료!")
    st.rerun()


# ------------------------------------------------------------------
# 단계별 카드 헬퍼
# ------------------------------------------------------------------
def step_header(no: int):
    stt = steps.get(no, {}).get("status", "pending")
    icon = {"done": "✅", "running": "⏳", "error": "❌"}.get(stt, "⚪")
    st.subheader(f"{icon} {no}단계 · {pipeline.STEP_NAMES[no]}")
    st.caption(pipeline.STEP_DESC[no])


def item_table(df: pd.DataFrame, cols_pref: list):
    show = [c for c in cols_pref if c in df.columns]
    st.dataframe(df[show] if show else df, width="stretch", hide_index=True)


_LEVEL_COLOR = {0: "#ff7043", 1: "#42a5f5", 2: "#66bb6a", 3: "#ab47bc"}  # 시드/1/2/3차수


def render_network(nodes: list, edges: list) -> None:
    """연관 네트워크를 plotly+networkx 스프링 레이아웃으로 컴팩트하게 그린다."""
    G = nx.Graph()
    label2id = {}
    for n in nodes:
        nid = str(n.get("id"))
        G.add_node(nid)
        if n.get("label"):
            label2id[str(n["label"])] = nid
    for e in edges:
        s = label2id.get(str(e.get("from")))
        t = label2id.get(str(e.get("to")))
        if s and t and s != t:
            G.add_edge(s, t)

    n_count = G.number_of_nodes()
    pos = nx.spring_layout(G, seed=42, k=1.6 / (n_count ** 0.5 or 1), iterations=60)
    meta = {str(n.get("id")): n for n in nodes}

    edge_x, edge_y = [], []
    for a, b in G.edges():
        edge_x += [pos[a][0], pos[b][0], None]
        edge_y += [pos[a][1], pos[b][1], None]
    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines",
                            line=dict(width=0.6, color="#cfd8dc"), hoverinfo="none")

    node_x, node_y, text, hover, color, size = [], [], [], [], [], []
    big = n_count <= 25  # 노드 적을 때만 라벨 표시
    for nid in G.nodes():
        m = meta.get(nid, {})
        lv = m.get("level", 1) or 0
        node_x.append(pos[nid][0]); node_y.append(pos[nid][1])
        lbl = str(m.get("label", nid))
        seed = bool(m.get("seed"))
        hover.append(f"{lbl}<br>{('시드' if seed else str(lv)+'차수')}")
        text.append(lbl if (seed or big) else "")
        color.append(_LEVEL_COLOR.get(0 if seed else lv, "#90a4ae"))
        size.append(26 if seed else (16 if lv == 1 else 10))
    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers+text", text=text, textposition="top center",
        textfont=dict(size=10), hovertext=hover, hoverinfo="text",
        marker=dict(size=size, color=color, line=dict(width=1, color="#ffffff")))

    fig = go.Figure([edge_trace, node_trace])
    fig.update_layout(
        height=520, showlegend=False, margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width="stretch")


# ==================================================================
# 🕸 직접 수집 (위키 크롤링) 모드 — APOLLO 단계 UI 대신 이 흐름을 렌더
# ==================================================================
if cfg.get("kind") == "crawl":
    seed_text = cfg.get("seed_text", "")
    st.subheader(f"🕸 직접 수집 — {seed_text}")
    st.caption(f"위키피디아/XTools를 직접 크롤링해 수집→적재→분석→AI 보고서까지 실행합니다. "
               f"(확장 차수 n={cfg.get('n_depth', 1)})")

    has_result = store.has_artifact(run_id, "summary")
    btn_label = "▶ 수집 실행 (수 분 소요)" if not has_result else "🔄 다시 수집"
    if st.button(btn_label, type="primary"):
        if crawl.crawl_busy():
            st.info("⏳ 지금 다른 사용자가 수집 중입니다. 순서가 되면 자동으로 시작됩니다 "
                    "(한 번에 한 명씩 수집).")
        bar = st.progress(0, text="시작…")
        log_area = st.empty()
        logs: list = []

        def _push(line: str):
            logs.append(line)
            log_area.code("\n".join(logs[-25:]))  # 최근 25줄 스크롤 로그

        def _cprog(s, t, m):
            name = crawl.CRAWL_STEP_NAMES.get(s, "")
            bar.progress(int(s / t * 100), text=f"[{s}/{t}] {name}")
            _push(f"[{s}/{t}] {name} — {m}")

        def _cdetail(ev, data):
            _push(f"    · {crawl.fmt_detail(ev, data)}")

        try:
            with st.spinner("크롤링 중… (한 번에 한 명씩 · XTools 수집이 가장 오래 걸립니다)"):
                crawl.run_crawl(run_id, seed_text, cfg.get("n_depth", 1),
                                progress=_cprog, detail=_cdetail)
            bar.progress(100, text="완료")
            st.success("수집 완료!")
        except Exception as exc:
            store.set_run_status(run_id, "error")
            st.error(f"수집 실패: {exc}")
        st.rerun()

    if has_result:
        seed = store.load_json(run_id, "seed", {})
        st.success(f"🌱 시드: **{seed.get('itemName')}**")
        if seed.get("url"):
            st.caption(f"🔗 {seed.get('url')}")
        if seed.get("descriptionKor"):
            st.caption(str(seed["descriptionKor"])[:400] + "…")

        st.markdown("#### 📊 수집 지표 종합 (기술집약도·수요/공급 부상성)")
        summ = store.load_df(run_id, "summary")
        item_table(summ, ["순위", "itemName", "score", "demandEmergence",
                           "supplyEmergence", "확산성", "종합점수"])

        net = store.load_json(run_id, "network", {})
        if net.get("nodes"):
            st.markdown(f"#### 🕸 위키 연관 네트워크 (노드 {len(net['nodes'])} · 엣지 {len(net['edges'])})")
            render_network(net["nodes"], net["edges"])
            st.caption("🟠 시드 · 🔵 연관 아이템 — 노드 hover로 제목 확인, 스크롤로 확대/축소")

        rep = store.get_report(run_id) or {}
        if rep.get("content"):
            st.markdown("#### 📄 AI 보고서")
            if rep.get("model") == "fallback":
                st.warning("LLM 미연결 — 지표 기반 기본 보고서입니다.")
            else:
                st.caption(f"모델: {rep.get('model')}")
            st.markdown(rep["content"])
            st.download_button("📥 보고서 다운로드 (.md)", rep["content"],
                               file_name=f"crawl_{run_id}.md", mime="text/markdown")
    st.stop()


# ============================ 1단계 ============================
step_header(1)
if st.button("▶ 1단계 실행", key="b1"):
    with st.spinner("TOP100 수집…"):
        pipeline.step1_seed(run_id, cfg["category"], cfg["indicator"], cfg["seed_rank"])
    st.rerun()

if _done(steps, 1):
    seed = store.load_json(run_id, "seed", {})
    st.success(f"🌱 시드 확정: **{seed.get('itemName')}**  (itemId {seed.get('itemId')})")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric(apollo.INDICATORS.get(cfg["indicator"], "점수"), seed.get("score"))
    with c2:
        if seed.get("descriptionKor"):
            st.caption(str(seed["descriptionKor"])[:400] + "…")
    top = store.load_json(run_id, "top100", {}).get("items", [])
    with st.expander(f"TOP100 원본 ({len(top)}건)"):
        item_table(pd.DataFrame(top), ["rank", "itemId", "itemName", "score", "descriptionKor"])
st.divider()

# ============================ 2단계 ============================
step_header(2)
if st.button("▶ 2단계 실행", key="b2", disabled=not _done(steps, 1)):
    with st.spinner("APOLLO 연관 네트워크 확장…"):
        pipeline.step2_expand(run_id, degree=cfg.get("net_degree", 3))
    st.rerun()

if _done(steps, 2):
    net = store.load_json(run_id, "network", {})
    exp = store.load_df(run_id, "expand")
    deg_ui = cfg.get("net_degree", 3) - 1
    st.info(f"연관 아이템 **{len(exp)}건** · 네트워크 노드 {len(net.get('nodes', []))} / "
            f"엣지 {len(net.get('edges', []))} · 확장 **{deg_ui}차수**")
    if exp.empty:
        st.warning("이 시드는 연관 네트워크가 비어 있습니다. 다른 데모 사례를 선택하세요.")
    else:
        item_table(exp, ["itemName", "category", "level"])
st.divider()

# ============================ 3단계 ============================
step_header(3)
if st.button("▶ 3단계 실행", key="b3", disabled=not _done(steps, 2)):
    with st.spinner("연관 아이템 지표 수집 (network/list 1콜)…"):
        pipeline.step3_details(run_id, degree=cfg.get("net_degree", 3))
    st.rerun()

if _done(steps, 3):
    det = store.load_df(run_id, "details")
    st.info(f"연관 아이템 **{len(det)}건**의 지표를 1회 호출로 수집 (기술집약도·수요/공급 부상도)")
    item_table(det, ["rank", "itemName", "category", "techIntensity",
                     "demandEmergence", "supplyEmergence"])
st.divider()

# ============================ 4단계 ============================
step_header(4)
if st.button("▶ 4단계 실행", key="b4", disabled=not _done(steps, 3)):
    with st.spinner("네트워크 그래프 준비…"):
        pipeline.step4_network(run_id)
    st.rerun()

if _done(steps, 4):
    net = store.load_json(run_id, "network", {})
    nodes = net.get("nodes", [])
    edges = net.get("edges", [])
    st.info(f"노드 **{len(nodes)}** · 엣지 **{len(edges)}**")
    if nodes:
        render_network(nodes, edges)
        st.caption("🟠 시드 · 🔵 1차수 이웃 · 🟢 2차수 이웃 — 노드에 마우스를 올리면 이름이 보이고, 스크롤로 확대/축소됩니다.")
    else:
        st.caption("네트워크 데이터가 없습니다.")
st.divider()

# ============================ 5단계 ============================
step_header(5)
if st.button("▶ 5단계 실행", key="b5", disabled=not _done(steps, 4)):
    with st.spinner("종합 스코어링…"):
        pipeline.step5_summary(run_id)
    st.rerun()

if _done(steps, 5):
    summ = store.load_df(run_id, "summary")
    item_table(summ, ["순위", "itemName", "category", "techIntensity",
                      "demandEmergence", "supplyEmergence", "종합점수"])
    if {"demandEmergence", "supplyEmergence"}.issubset(summ.columns) and not summ.empty:
        plot = summ.dropna(subset=["demandEmergence", "supplyEmergence"])
        if not plot.empty:
            st.markdown("**🗺️ 수요·공급 부상도 맵** (오른쪽 위일수록 유망)")
            has_score = "종합점수" in plot.columns
            fig = px.scatter(plot, x="supplyEmergence", y="demandEmergence",
                             size="종합점수" if has_score else None, text="itemName",
                             color="종합점수" if has_score else None,
                             color_continuous_scale="Turbo",
                             labels={"supplyEmergence": "공급 부상도", "demandEmergence": "수요 부상도"})
            fig.update_traces(textposition="top center")
            fig.update_layout(height=460)
            st.plotly_chart(fig, width="stretch")
st.divider()

# ============================ 6단계 ============================
step_header(6)
if st.button("▶ 6단계 실행 (AI 보고서)", key="b6", disabled=not _done(steps, 5)):
    with st.spinner("AI 보고서 생성…"):
        pipeline.step6_report(run_id)
    st.rerun()

if _done(steps, 6):
    rep = store.get_report(run_id) or {}
    if rep.get("model") == "fallback":
        st.warning("LLM 미연결 — 지표 기반 기본 보고서입니다.")
    else:
        st.caption(f"모델: {rep.get('model')}")
    st.markdown(rep.get("content", ""))
    st.download_button("📥 보고서 다운로드 (.md)", rep.get("content", ""),
                       file_name=f"report_{run_id}.md", mime="text/markdown")