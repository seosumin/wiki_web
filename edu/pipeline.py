# -*- coding: utf-8 -*-
"""
교육용 '유망아이템 발굴' APOLLO 파이프라인 오케스트레이터 (6단계).

기존 위키/XTools 크롤링을 KISTI APOLLO A4 공식 API로 대체.
교육생이 '다음 단계' 버튼으로 하나씩 실행/관찰. run_all()은 강사 시연/데모 캐싱용.

단계 (재설계 — 시드의 연관 네트워크를 API로 확장하고 그 아이템들을 분석)
  1 시드 아이템 확정   ← A4.1 TOP100에서 카테고리·지표별 시드 선택
  2 연관 네트워크 확장 ← A4.4 시드의 network(nodes/edges) 확보 (연관 아이템 = 노드)
  3 아이템 상세 수집   ← A4.3 각 네트워크 노드의 details(수요·공급 부상도, 추세)
  4 네트워크 그래프 시각화 ← 2단계 네트워크를 3단계 지표로 강조해 표시
  5 종합 스코어링      ← 지표 병합 랭킹 테이블
  6 AI 보고서 생성     ← LLM 요약/해석

데이터 주의 (테스트 서버 실측):
  - TECH_INTENSITY + 인공지능/반도체 시드는 A4.4 네트워크가 풍부(수십~97노드).
  - A4.4 노드 id는 A4.3 details가 대체로 동작(일부 400 → 건너뜀).
  - 시드/카테고리에 따라 네트워크가 비어있을 수 있어 자유 입력 대신 데모 프리셋 사용.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from . import apollo, store

TOTAL_STEPS = 6

STEP_NAMES: Dict[int, str] = {
    1: "시드 아이템 확정",
    2: "연관 네트워크 확장",
    3: "아이템 상세 수집",
    4: "네트워크 그래프 시각화",
    5: "종합 스코어링",
    6: "AI 보고서 생성",
}

STEP_DESC: Dict[int, str] = {
    1: "국가전략기술 카테고리·지표로 APOLLO TOP100을 불러와 분석 출발점(시드)을 확정합니다.",
    2: "확정된 시드를 APOLLO 연관 네트워크 API로 확장해, 함께 부상하는 연관 아이템들을 끌어옵니다.",
    3: "확장된 연관 아이템의 지표(기술집약도·수요/공급 부상도)를 APOLLO network/list API로 한 번에 수집합니다.",
    4: "확장된 네트워크를 그래프로 시각화합니다(수집한 지표로 노드 강조).",
    5: "수집한 지표를 하나의 랭킹 표로 종합해 유망도를 정렬합니다.",
    6: "종합 결과를 LLM이 읽고 유망아이템 발굴 보고서를 자동 작성합니다.",
}

# 데모 프리셋: A4.4 네트워크가 풍부하게 잡히는(에러 없는) 조합만 큐레이션.
# (2026-07-21 12개 카테고리 스캔 — TECH_INTENSITY에서 노드 70~97개 확보되는 사례만 선별.
#  이차전지·원자력·수소·사이버보안·통신·로봇·모빌리티는 테스트 서버 TOP100 미제공)
DEMO_PRESETS: List[Dict[str, Any]] = [
    {"category": "인공지능", "indicator": "TECH_INTENSITY", "seed_rank": 1,
     "label": "인공지능 · 기술집약도 1위 (Mathematical finance)"},
    {"category": "양자", "indicator": "TECH_INTENSITY", "seed_rank": 1,
     "label": "양자 · 기술집약도 1위 (Diffraction)"},
    {"category": "반도체·디스플레이", "indicator": "TECH_INTENSITY", "seed_rank": 3,
     "label": "반도체·디스플레이 · 기술집약도 3위 (Graphics processing unit)"},
    {"category": "우주항공·해양", "indicator": "TECH_INTENSITY", "seed_rank": 1,
     "label": "우주항공·해양 · 기술집약도 1위 (Ship)"},
    {"category": "첨단바이오", "indicator": "TECH_INTENSITY", "seed_rank": 2,
     "label": "첨단바이오 · 기술집약도 2위 (Population genetics)"},
]

ProgressCb = Optional[Callable[[int, int, str], None]]


def _emit(cb: ProgressCb, step: int, msg: str) -> None:
    if cb:
        cb(step, TOTAL_STEPS, msg)


def _num(x: Any) -> Optional[float]:
    try:
        return round(float(x), 3)
    except (TypeError, ValueError):
        return None


# ============================================================
# 1) 시드 아이템 확정  (A4.1 TOP100)
# ============================================================
def step1_seed(run_id: str, category: str, indicator: str = "TECH_INTENSITY",
               seed_rank: int = 1, progress: ProgressCb = None) -> Dict[str, Any]:
    _emit(progress, 1, f"'{category}' TOP100 불러오는 중…")
    store.start_step(run_id, 1, STEP_NAMES[1])

    top = apollo.top100(category, indicator)
    if not top:
        store.finish_step(run_id, 1, 0, status="error")
        raise apollo.ApolloError(f"'{category}' TOP100 결과가 비어 있습니다.")

    store.save_json(run_id, "top100", {"category": category, "indicator": indicator, "items": top})

    idx = max(0, min(seed_rank - 1, len(top) - 1))
    seed = top[idx]
    seed_id = int(seed.get("itemId"))
    seed_name = str(seed.get("itemName", ""))
    store.set_run_item(run_id, seed_id, seed_name, category)
    store.save_json(run_id, "seed", {**seed, "category": category, "indicator": indicator})

    store.finish_step(run_id, 1, len(top))
    _emit(progress, 1, f"시드 확정: {seed_name} (id={seed_id})")
    return seed


# ============================================================
# 2) 연관 네트워크 확장  (A4.4 item_network)
# ============================================================
def step2_expand(run_id: str, degree: int = 3, progress: ProgressCb = None) -> pd.DataFrame:
    """시드의 APOLLO 연관 네트워크를 확보. 노드(연관 아이템)+엣지 저장."""
    _emit(progress, 2, "APOLLO 연관 네트워크 확장 중…")
    store.start_step(run_id, 2, STEP_NAMES[2])

    seed = store.load_json(run_id, "seed", {})
    seed_id = seed.get("itemId")
    seed_label = str(seed.get("itemName", ""))
    if seed_id is None:
        store.finish_step(run_id, 2, 0, status="error")
        raise apollo.ApolloError("시드가 없습니다. 1단계를 먼저 실행하세요.")

    net = apollo.item_network(int(seed_id), degree=degree)
    raw_nodes = net.get("nodes", []) or []
    raw_edges = net.get("edges", []) or []

    nodes = [{
        "id": str(n.get("id")),
        "label": n.get("label"),
        "category": n.get("category"),
        "level": n.get("level", 1),
        "seed": (n.get("label") == seed_label) or (str(n.get("id")) == str(seed_id)),
    } for n in raw_nodes]
    edges = [{"from": e.get("from"), "to": e.get("to")} for e in raw_edges]
    graph = {"nodes": nodes, "edges": edges, "source": "network", "degree": degree}
    store.save_json(run_id, "network", graph)

    # 연관 아이템 목록(상세 수집 대상) — itemId 보유 노드
    rows = [{"itemId": n["id"], "itemName": n["label"], "category": n["category"],
             "level": n["level"], "seed": n["seed"]}
            for n in nodes if n.get("id") not in (None, "None")]
    df = pd.DataFrame(rows)
    store.save_df(run_id, "expand", df)

    store.finish_step(run_id, 2, len(nodes))
    _emit(progress, 2, f"네트워크 확장: 노드 {len(nodes)} / 엣지 {len(edges)}")
    if not nodes:
        # 데모 프리셋 외 시드는 네트워크가 비어있을 수 있음
        _emit(progress, 2, "⚠️ 네트워크가 비어 있습니다 (이 시드는 연관망이 없거나 API 미지원).")
    return df


# ============================================================
# 3) 아이템 상세 수집  (A4.5 network/list — 연관 아이템 지표를 한 번에)
# ============================================================
def step3_details(run_id: str, degree: int = 3, progress: ProgressCb = None) -> pd.DataFrame:
    """
    시드의 연관 아이템 지표를 A4.5 network/list로 '한 번의 호출'로 수집.
    각 항목: 기술집약도/수요부상도/공급부상도 3지표 포함(itemId 없음).
    (기존엔 아이템마다 A4.3 details를 호출해 느렸음 → 1콜로 대체)
    """
    _emit(progress, 3, "연관 아이템 지표 수집 중… (network/list 1회 호출)")
    store.start_step(run_id, 3, STEP_NAMES[3])

    seed = store.load_json(run_id, "seed", {})
    seed_id = seed.get("itemId")
    if seed_id is None:
        store.finish_step(run_id, 3, 0, status="error")
        raise apollo.ApolloError("시드가 없습니다. 1단계를 먼저 실행하세요.")

    lst = apollo.network_list(int(seed_id), degree=degree)
    rows = [{
        "rank": it.get("rank"),
        "itemName": it.get("itemName"),
        "category": it.get("category"),
        "techIntensity": _num(it.get("techIntensity")),
        "demandEmergence": _num(it.get("demandEmergence")),
        "supplyEmergence": _num(it.get("supplyEmergence")),
    } for it in lst]
    out = pd.DataFrame(rows)

    store.save_df(run_id, "details", out)
    store.finish_step(run_id, 3, len(out))
    _emit(progress, 3, f"연관 아이템 지표 {len(out)}건 수집 완료 (1콜)")
    return out


# ============================================================
# 4) 네트워크 그래프 시각화  (2단계 네트워크 + 3단계 지표)
# ============================================================
def step4_network(run_id: str, progress: ProgressCb = None) -> Dict[str, Any]:
    """네트워크 노드에 수집 지표를 붙여 시각화용으로 강화(노드 크기/색)."""
    _emit(progress, 4, "네트워크 그래프 준비 중…")
    store.start_step(run_id, 4, STEP_NAMES[4])

    graph = store.load_json(run_id, "network", {"nodes": [], "edges": []})
    det = store.load_df(run_id, "details")
    # network/list 항목엔 itemId가 없어 itemName(=그래프 노드 label)으로 매칭
    score_by_name = {}
    if not det.empty:
        for _, r in det.iterrows():
            vals = [float(r[k]) for k in ("techIntensity", "demandEmergence", "supplyEmergence")
                    if k in det.columns and pd.notna(r.get(k))]
            if vals:
                score_by_name[str(r.get("itemName"))] = round(sum(vals) / len(vals), 2)

    for n in graph.get("nodes", []):
        n["score"] = score_by_name.get(str(n.get("label")))
    store.save_json(run_id, "network", graph)

    store.finish_step(run_id, 4, len(graph.get("nodes", [])))
    _emit(progress, 4, f"네트워크 그래프 준비 완료 (노드 {len(graph.get('nodes', []))})")
    return graph


# ============================================================
# 5) 종합 스코어링  (랭킹 테이블)
# ============================================================
def step5_summary(run_id: str, progress: ProgressCb = None) -> pd.DataFrame:
    _emit(progress, 5, "종합 스코어링 중…")
    store.start_step(run_id, 5, STEP_NAMES[5])

    df = store.load_df(run_id, "details")
    if not df.empty:
        cols = [c for c in ("techIntensity", "demandEmergence", "supplyEmergence") if c in df.columns]
        if cols:
            comp = pd.concat([df[c].fillna(0) for c in cols], axis=1).mean(axis=1)
            df = df.copy()
            df["종합점수"] = comp.round(2)
            df = df.sort_values("종합점수", ascending=False).reset_index(drop=True)
            df.insert(0, "순위", range(1, len(df) + 1))

    store.save_df(run_id, "summary", df)
    store.finish_step(run_id, 5, len(df))
    _emit(progress, 5, f"종합 랭킹 {len(df)}건 완성")
    return df


# ============================================================
# 6) AI 보고서 생성  (edu.llm)
# ============================================================
def step6_report(run_id: str, progress: ProgressCb = None) -> str:
    _emit(progress, 6, "AI 보고서 생성 중…")
    store.start_step(run_id, 6, STEP_NAMES[6])

    from . import llm  # 지연 임포트 (openai 의존)
    seed = store.load_json(run_id, "seed", {})
    summary = store.load_df(run_id, "summary")
    content, model = llm.generate_report(seed, summary)

    store.save_report(run_id, content, model)
    store.finish_step(run_id, 6, len(summary))
    store.set_run_status(run_id, "done")
    _emit(progress, 6, f"보고서 생성 완료 (모델: {model})")
    return content


# ============================================================
# 전체 실행 (강사 시연 / 데모 캐싱)
# ============================================================
def run_all(run_id: str, category: str, indicator: str = "TECH_INTENSITY",
            seed_rank: int = 1, net_degree: int = 3,
            progress: ProgressCb = None) -> Dict[str, Any]:
    step1_seed(run_id, category, indicator, seed_rank, progress)
    step2_expand(run_id, degree=net_degree, progress=progress)
    step3_details(run_id, degree=net_degree, progress=progress)
    step4_network(run_id, progress=progress)
    summary = step5_summary(run_id, progress=progress)
    report = step6_report(run_id, progress=progress)
    return {"run_id": run_id, "summary": summary, "report": report}
