# -*- coding: utf-8 -*-
"""
스토리지 계층 — 엑셀 파일 입출력을 대체하는 DB 계층.

- Supabase(Postgres) 연결 문자열이 있으면 그걸, 없으면 로컬 SQLite로 폴백.
- 모든 데이터는 run_id 로 격리 → 교육생 다중 사용자 자연 분리.
- 단계별 산출물(DataFrame/dict/list)은 JSON 페이로드로 범용 저장.

테이블
  runs        : 분석 실행 1건 (사용자/시드/아이템/상태)
  run_steps   : 실행의 7단계별 상태·행수·시각
  step_data   : (run_id, artifact) → JSON 페이로드 (단계 산출물)
  reports     : (run_id) → LLM 보고서 텍스트
"""
from __future__ import annotations

import datetime
import json
import uuid
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy import (
    JSON, Column, DateTime, Integer, MetaData, String, Table, Text,
    create_engine, delete, insert, select, update,
)
from sqlalchemy.engine import Engine

from . import config

_engine: Optional[Engine] = None
_metadata = MetaData()

runs = Table(
    "edu_runs", _metadata,
    Column("run_id", String(40), primary_key=True),
    Column("user_name", String(120)),
    Column("seed_query", String(300)),
    Column("item_id", Integer),
    Column("item_name", String(300)),
    Column("category", String(120)),
    Column("mode", String(20)),            # demo | live
    Column("status", String(20)),          # running | done | error
    Column("created_at", DateTime),
)

run_steps = Table(
    "edu_run_steps", _metadata,
    Column("run_id", String(40), primary_key=True),
    Column("step_no", Integer, primary_key=True),
    Column("name", String(120)),
    Column("status", String(20)),          # pending | running | done | error
    Column("row_count", Integer),
    Column("started_at", DateTime),
    Column("finished_at", DateTime),
)

step_data = Table(
    "edu_step_data", _metadata,
    Column("run_id", String(40), primary_key=True),
    Column("artifact", String(80), primary_key=True),
    Column("payload", JSON),
    Column("updated_at", DateTime),
)

reports = Table(
    "edu_reports", _metadata,
    Column("run_id", String(40), primary_key=True),
    Column("content", Text),
    Column("model", String(120)),
    Column("created_at", DateTime),
)


DB_BACKEND = "unknown"  # 실제 사용 중인 백엔드: "postgres" | "sqlite"


def _make_engine(url: str) -> Engine:
    connect_args: Dict[str, Any] = {}
    kwargs: Dict[str, Any] = {"future": True, "pool_pre_ping": True}
    if url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
    eng = create_engine(url, connect_args=connect_args, **kwargs)
    # 실제 연결 시도 (paused Supabase 등 조기 감지)
    with eng.connect():
        pass
    return eng


def get_engine() -> Engine:
    """설정된 DB(Supabase 등) 연결. 실패하면 로컬 SQLite로 자동 폴백."""
    global _engine, DB_BACKEND
    if _engine is None:
        url = config.db_url()
        try:
            _engine = _make_engine(url)
            DB_BACKEND = "sqlite" if url.startswith("sqlite") else "postgres"
        except Exception as exc:
            if url.startswith("sqlite"):
                raise
            sqlite_url = f"sqlite:///{(config.PROJECT_DIR / 'edu.db').as_posix()}"
            print(f"[store] DB 연결 실패({url.split('@')[-1][:40]}…) → SQLite 폴백: {exc.__class__.__name__}")
            _engine = _make_engine(sqlite_url)
            DB_BACKEND = "sqlite"
        _metadata.create_all(_engine)
    return _engine


def _now() -> datetime.datetime:
    return datetime.datetime.now()


def _upsert(table: Table, pk_cols: List[str], values: Dict[str, Any]) -> None:
    """DB 무관 간단 upsert (존재하면 update, 아니면 insert)."""
    eng = get_engine()
    with eng.begin() as conn:
        cond = None
        for c in pk_cols:
            clause = table.c[c] == values[c]
            cond = clause if cond is None else (cond & clause)
        exists = conn.execute(select(table.c[pk_cols[0]]).where(cond)).first()
        if exists:
            upd = {k: v for k, v in values.items() if k not in pk_cols}
            conn.execute(update(table).where(cond).values(**upd))
        else:
            conn.execute(insert(table).values(**values))


# ============================================================
# run 라이프사이클
# ============================================================
def create_run(user_name: str, seed_query: str, mode: str = "live") -> str:
    run_id = uuid.uuid4().hex[:16]
    eng = get_engine()
    with eng.begin() as conn:
        conn.execute(insert(runs).values(
            run_id=run_id, user_name=user_name or "guest", seed_query=seed_query,
            item_id=None, item_name=None, category=None,
            mode=mode, status="running", created_at=_now(),
        ))
    return run_id


def set_run_item(run_id: str, item_id: int, item_name: str, category: str) -> None:
    eng = get_engine()
    with eng.begin() as conn:
        conn.execute(update(runs).where(runs.c.run_id == run_id).values(
            item_id=item_id, item_name=item_name, category=category))


def set_run_status(run_id: str, status: str) -> None:
    eng = get_engine()
    with eng.begin() as conn:
        conn.execute(update(runs).where(runs.c.run_id == run_id).values(status=status))


def get_run(run_id: str) -> Optional[Dict[str, Any]]:
    eng = get_engine()
    with eng.begin() as conn:
        row = conn.execute(select(runs).where(runs.c.run_id == run_id)).mappings().first()
    return dict(row) if row else None


def list_runs(user_name: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
    eng = get_engine()
    q = select(runs).order_by(runs.c.created_at.desc()).limit(limit)
    if user_name:
        q = q.where(runs.c.user_name == user_name)
    with eng.begin() as conn:
        rows = conn.execute(q).mappings().all()
    return [dict(r) for r in rows]


# ============================================================
# 단계 상태
# ============================================================
def start_step(run_id: str, step_no: int, name: str) -> None:
    _upsert(run_steps, ["run_id", "step_no"], dict(
        run_id=run_id, step_no=step_no, name=name, status="running",
        row_count=0, started_at=_now(), finished_at=None))


def finish_step(run_id: str, step_no: int, row_count: int = 0, status: str = "done") -> None:
    _upsert(run_steps, ["run_id", "step_no"], dict(
        run_id=run_id, step_no=step_no, name=_step_name(run_id, step_no),
        status=status, row_count=row_count, started_at=None, finished_at=_now()))


def _step_name(run_id: str, step_no: int) -> str:
    eng = get_engine()
    with eng.begin() as conn:
        row = conn.execute(select(run_steps.c.name).where(
            (run_steps.c.run_id == run_id) & (run_steps.c.step_no == step_no))).first()
    return row[0] if row else ""


def get_steps(run_id: str) -> List[Dict[str, Any]]:
    eng = get_engine()
    with eng.begin() as conn:
        rows = conn.execute(select(run_steps).where(run_steps.c.run_id == run_id)
                            .order_by(run_steps.c.step_no)).mappings().all()
    return [dict(r) for r in rows]


# ============================================================
# 단계 산출물 (엑셀 to_excel/read_excel 대체)
# ============================================================
def save_json(run_id: str, artifact: str, payload: Any) -> None:
    _upsert(step_data, ["run_id", "artifact"], dict(
        run_id=run_id, artifact=artifact, payload=payload, updated_at=_now()))


def load_json(run_id: str, artifact: str, default: Any = None) -> Any:
    eng = get_engine()
    with eng.begin() as conn:
        row = conn.execute(select(step_data.c.payload).where(
            (step_data.c.run_id == run_id) & (step_data.c.artifact == artifact))).first()
    if not row:
        return default
    val = row[0]
    # SQLite JSON 컬럼은 문자열로 돌아올 수 있음
    if isinstance(val, str):
        try:
            return json.loads(val)
        except Exception:
            return val
    return val


def save_df(run_id: str, artifact: str, df: pd.DataFrame) -> None:
    """DataFrame → JSON(records + columns)로 저장."""
    payload = {
        "columns": list(df.columns),
        "records": json.loads(df.to_json(orient="records", force_ascii=False)),
    }
    save_json(run_id, artifact, payload)


def load_df(run_id: str, artifact: str) -> pd.DataFrame:
    payload = load_json(run_id, artifact)
    if not payload or "records" not in payload:
        return pd.DataFrame()
    df = pd.DataFrame(payload["records"])
    cols = payload.get("columns")
    if cols:
        df = df.reindex(columns=cols)
    return df


def has_artifact(run_id: str, artifact: str) -> bool:
    eng = get_engine()
    with eng.begin() as conn:
        row = conn.execute(select(step_data.c.artifact).where(
            (step_data.c.run_id == run_id) & (step_data.c.artifact == artifact))).first()
    return row is not None


# ============================================================
# LLM 보고서
# ============================================================
def save_report(run_id: str, content: str, model: str) -> None:
    _upsert(reports, ["run_id"], dict(
        run_id=run_id, content=content, model=model, created_at=_now()))


def get_report(run_id: str) -> Optional[Dict[str, Any]]:
    eng = get_engine()
    with eng.begin() as conn:
        row = conn.execute(select(reports).where(reports.c.run_id == run_id)).mappings().first()
    return dict(row) if row else None


# ============================================================
# 데모 캐시: 같은 (item_id) 데모 실행이 있으면 복제 없이 재사용용 조회
# ============================================================
def find_demo_run(item_id: int) -> Optional[str]:
    eng = get_engine()
    with eng.begin() as conn:
        row = conn.execute(select(runs.c.run_id).where(
            (runs.c.item_id == item_id) & (runs.c.mode == "demo") &
            (runs.c.status == "done")).order_by(runs.c.created_at.desc())).first()
    return row[0] if row else None
