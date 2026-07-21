# -*- coding: utf-8 -*-
"""
'직접 수집' 모드 — 원본 위키피디아/XTools 크롤링 파이프라인(project root의 pipeline.py)을
교육툴 store/보고서 계층에 연결한다.

APOLLO 모드가 공식 API로 빠르게 도는 반면, 이 모드는 교육생이 원하는 아이템명을
자유롭게 입력해 '수집→적재→분석→보고서' 전 과정을 직접 체험하도록 한다.
크롤링(XTools)이 병목이라 느리므로 심화/시연용.

주의:
- Windows Streamlit 프로세스에서 실행(원본 파이프라인이 Windows/venv 기준).
- wiki_crawling 은 verify=False + User-Agent 세션이라 KISTI망 TLS 가로채기를 통과.
- 통계 지표(기술집약도/공급부상성/수요부상성)가 edu 보고서 스키마와 그대로 대응.
"""
from __future__ import annotations

import os
import threading
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

import pipeline as wiki_pipeline  # project root (app.py가 sys.path에 루트 추가)

from . import llm, store

# 크롤링 직렬화 락 — 한 번에 1건만 수집.
# 이유: ①XTools rate-limit(동시 요청 폭주 시 429→다같이 느려짐) ②같은 아이템 동시
# 크롤 시 runs/<item>/ 파일 충돌 방지. 단일 Streamlit 프로세스 내 전 세션이 공유.
_CRAWL_LOCK = threading.Lock()


def crawl_busy() -> bool:
    """다른 세션이 지금 크롤링 중인지(락 점유) 여부."""
    if _CRAWL_LOCK.acquire(blocking=False):
        _CRAWL_LOCK.release()
        return False
    return True


# --- 백그라운드 작업(job) 관리: 중지 버튼 지원 -----------------------
# Streamlit은 동기 크롤 중 스크립트가 블록돼 버튼을 못 받으므로,
# 크롤을 별도 스레드로 돌리고 stop Event를 콜백에서 확인해 협조적으로 중단한다.
_JOBS: Dict[str, Dict[str, Any]] = {}
_JOBS_LOCK = threading.Lock()


class CrawlStopped(BaseException):
    """사용자 중지. BaseException 상속 → 파이프라인의 except Exception에 안 걸리고 전파."""


def start_crawl_job(run_id: str, seed: str, n_depth: int = 1) -> None:
    """백그라운드 스레드로 크롤 시작. 진행/로그/상태를 _JOBS[run_id]에 기록."""
    stop = threading.Event()
    job: Dict[str, Any] = {"stop": stop, "logs": [], "step": (0, CRAWL_TOTAL, "대기 중…"),
                           "status": "running", "error": None}
    with _JOBS_LOCK:
        _JOBS[run_id] = job

    def _prog(s, t, m):
        if stop.is_set():
            raise CrawlStopped()
        job["step"] = (s, t, m)
        job["logs"].append(f"[{s}/{t}] {CRAWL_STEP_NAMES.get(s, '')} — {m}")

    def _det(ev, data):
        if stop.is_set():
            raise CrawlStopped()
        job["logs"].append(f"    · {fmt_detail(ev, data or {})}")

    def _worker():
        try:
            run_crawl(run_id, seed, n_depth, progress=_prog, detail=_det)
            job["status"] = "done"
        except CrawlStopped:
            job["status"] = "stopped"
            job["logs"].append("⛔ 사용자 중지")
            store.set_run_status(run_id, "error")
        except BaseException as exc:  # noqa
            job["status"] = "error"
            job["error"] = str(exc)
            store.set_run_status(run_id, "error")

    th = threading.Thread(target=_worker, daemon=True)
    job["thread"] = th
    th.start()


def stop_crawl_job(run_id: str) -> None:
    with _JOBS_LOCK:
        j = _JOBS.get(run_id)
    if j:
        j["stop"].set()


def get_job(run_id: str) -> Optional[Dict[str, Any]]:
    with _JOBS_LOCK:
        return _JOBS.get(run_id)

# 원본 파이프라인의 7단계 (진행 표시용)
CRAWL_STEP_NAMES: Dict[int, str] = {
    1: "시드 확인", 2: "네트워크 확장", 3: "네트워크 필터링",
    4: "XTools 수집", 5: "지표 통합", 6: "PageRank", 7: "통계 집계",
}
CRAWL_TOTAL = 7

ProgressCb = Optional[Callable[[int, int, str], None]]


def _read_excel(path: Optional[str]) -> pd.DataFrame:
    try:
        if path and os.path.isfile(path) and os.path.getsize(path) > 0:
            return pd.read_excel(path)
    except Exception:
        pass
    return pd.DataFrame()


def _num(x: Any) -> Optional[float]:
    try:
        return round(float(x), 4)
    except (TypeError, ValueError):
        return None


def _to_summary(stats: pd.DataFrame) -> pd.DataFrame:
    """통계 Excel → edu 종합 랭킹 스키마.
    (기술집약도→score, 수요부상성→demandEmergence, 공급부상성→supplyEmergence)
    """
    if stats is None or stats.empty:
        return pd.DataFrame()
    df = pd.DataFrame({
        "itemName": stats.get("title"),
        "score": stats.get("기술집약도"),
        "demandEmergence": stats.get("수요부상성"),
        "supplyEmergence": stats.get("공급부상성"),
        "확산성": stats.get("확산성"),
    })
    parts = [df[c].fillna(0) for c in ["score", "demandEmergence", "supplyEmergence"] if c in df]
    if parts:
        comp = pd.concat(parts, axis=1).mean(axis=1)
        df["종합점수"] = comp.round(4)
        df = df.sort_values("종합점수", ascending=False).reset_index(drop=True)
    df.insert(0, "순위", range(1, len(df) + 1))
    return df


def _to_graph(net: Dict[str, List], seed_true: str) -> Dict[str, Any]:
    """format_item_3_network({nodes:[{id,group,size}], edges:[{source,target}]})
    → render_network 스키마({nodes:[{id,label,level,seed}], edges:[{from,to}]})."""
    nodes = []
    for n in net.get("nodes", []):
        nid = str(n.get("id"))
        is_seed = (nid == seed_true)
        nodes.append({"id": nid, "label": nid, "category": "위키",
                      "level": 0 if is_seed else 1, "seed": is_seed})
    edges = [{"from": str(e.get("source")), "to": str(e.get("target"))}
             for e in net.get("edges", [])]
    return {"nodes": nodes, "edges": edges, "source": "wiki"}


def fmt_detail(event: str, data: Dict[str, Any]) -> str:
    """on_detail 이벤트를 사람이 읽는 로그 한 줄로."""
    d = data or {}
    if event == "collect_node":
        return f"수집 {d.get('idx', 0) + 1}/{d.get('total', '?')}: {d.get('node', '')}"
    if event == "collect_start":
        return f"XTools 수집 시작 — 노드 {len(d.get('nodes', []))}개"
    if event == "expand_done":
        return f"확장 완료 — 노드 {d.get('nodes', '?')} / 엣지 {d.get('edges', '?')}"
    if event == "filter_done":
        return f"필터 {d.get('before', '?')}→{d.get('after', '?')} (노드 {d.get('nodes', '?')})"
    if event == "pagerank_done":
        return f"PageRank 완료 — {d.get('total', '?')}개"
    if event == "stats_done":
        return f"통계 완료 — {d.get('total_items', '?')}개 아이템"
    return f"{event}"


def run_crawl(run_id: str, seed: str, n_depth: int = 1, progress: ProgressCb = None,
              detail: Optional[Callable[[str, Dict[str, Any]], None]] = None,
              lock_timeout: float = 1200) -> Dict[str, Any]:
    """위키 크롤링 파이프라인 실행 → edu store 적재 + LLM 보고서 생성.
    직렬화 락으로 한 번에 1건만 실행(앞선 수집이 있으면 lock_timeout초까지 대기)."""
    got = _CRAWL_LOCK.acquire(timeout=lock_timeout)
    if not got:
        raise TimeoutError("앞선 수집이 너무 오래 걸려 대기 시간을 초과했습니다. 잠시 후 다시 시도하세요.")
    try:
        return _run_crawl_locked(run_id, seed, n_depth, progress, detail)
    finally:
        _CRAWL_LOCK.release()


def _run_crawl_locked(run_id: str, seed: str, n_depth: int, progress: ProgressCb,
                      detail: Optional[Callable[[str, Dict[str, Any]], None]]) -> Dict[str, Any]:
    def on_prog(s: int, t: int, m: str) -> None:
        if progress:
            progress(s, t, m)

    def on_det(event: str, data: Dict[str, Any]) -> None:
        if detail:
            detail(event, data or {})

    paths, outs, seed_true, seed_url, seed_summary = wiki_pipeline.run_analysis_pipeline(
        seed, n_depth=n_depth, use_existing=True, on_progress=on_prog, on_detail=on_det)

    stats = _read_excel(paths.get("stats"))
    summary = _to_summary(stats)

    seed_dict = {
        "itemName": seed_true, "category": "위키 네트워크",
        "descriptionKor": (seed_summary or "")[:600], "url": seed_url, "score": None,
    }
    store.set_run_item(run_id, 0, seed_true, "위키 네트워크")
    store.save_json(run_id, "seed", seed_dict)
    store.save_df(run_id, "summary", summary)

    pr_df = _read_excel(paths.get("pagerank"))
    try:
        net_raw = wiki_pipeline.format_item_3_network(paths.get("filter"), pr_df)
    except Exception:
        net_raw = {"nodes": [], "edges": []}
    graph = _to_graph(net_raw, seed_true)
    store.save_json(run_id, "network", graph)

    if progress:
        progress(7, 7, "AI 보고서 생성 중…")
    content, model = llm.generate_report(seed_dict, summary)
    store.save_report(run_id, content, model)
    store.set_run_status(run_id, "done")

    return {"seed": seed_dict, "summary": summary, "graph": graph,
            "report": content, "model": model}
