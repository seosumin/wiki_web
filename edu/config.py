# -*- coding: utf-8 -*-
"""설정 로딩 — Streamlit secrets 또는 환경변수에서 읽고, 없으면 안전한 기본값."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

PROJECT_DIR = Path(__file__).resolve().parent.parent


def _secret(section: str, key: str) -> Optional[str]:
    """st.secrets[section][key] → 환경변수 순으로 조회. Streamlit 없이도 동작."""
    try:
        import streamlit as st  # 지연 임포트 (비-Streamlit 환경 허용)
        if section in st.secrets and key in st.secrets[section]:
            return str(st.secrets[section][key])
    except Exception:
        pass
    return os.environ.get(f"{section.upper()}_{key.upper()}")


def db_url() -> str:
    """DB 연결 문자열. Supabase 미설정 시 로컬 SQLite 폴백."""
    url = _secret("supabase", "db_url") or os.environ.get("DATABASE_URL")
    if url:
        # SQLAlchemy는 postgresql:// 스킴 필요 (postgres:// 는 구형)
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql://", 1)
        return url
    return f"sqlite:///{(PROJECT_DIR / 'edu.db').as_posix()}"


def is_sqlite() -> bool:
    return db_url().startswith("sqlite")


def llm_base_url() -> str:
    """로컬 LLM(OpenAI 호환) 엔드포인트. 기본값은 Ollama."""
    return _secret("llm", "base_url") or "http://localhost:11434/v1"


def llm_model() -> str:
    return _secret("llm", "model") or "qwen3:8b"


def llm_api_key() -> str:
    # Ollama/vLLM 은 보통 임의 키 허용
    return _secret("llm", "api_key") or "EMPTY"
