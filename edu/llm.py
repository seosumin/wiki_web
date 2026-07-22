# -*- coding: utf-8 -*-
"""
LLM 보고서 생성 — 종합 스코어링 결과를 읽어 '유망아이템 발굴 보고서'를 작성.

- 로컬 vLLM(OpenAI 호환) 엔드포인트를 requests로 직접 호출 → 'openai' 패키지 불필요.
- LLM 미가동/오류 시에도 데모가 멈추지 않도록 데이터 기반 폴백 보고서를 항상 제공.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

import pandas as pd

from . import config

# Qwen3 등 사고(thinking) 모델이 남기는 <think>…</think> 블록 제거용
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _clean(text: str) -> str:
    return _THINK_RE.sub("", text or "").strip()

SYSTEM_PROMPT = (
    "당신은 KISTI의 기술기획 분석가입니다. 주어진 유망아이템 지표 데이터를 근거로, "
    "교육생이 이해하기 쉬운 한국어 '유망아이템 발굴 보고서'를 작성합니다. "
    "데이터에 없는 수치를 지어내지 말고, 지표(기술집약도·수요부상도·공급부상도)의 "
    "의미를 풀어 해석하세요. 마크다운으로 간결하게 작성합니다."
)


def _rows_for_prompt(summary: pd.DataFrame, limit: int = 12) -> List[Dict[str, Any]]:
    if summary is None or summary.empty:
        return []
    keep = [c for c in ["순위", "itemName", "category", "score",
                        "demandEmergence", "supplyEmergence", "종합점수"] if c in summary.columns]
    return summary[keep].head(limit).to_dict(orient="records")


def _build_prompt(seed: Dict[str, Any], rows: List[Dict[str, Any]]) -> str:
    import json
    seed_line = f"- 시드 아이템: {seed.get('itemName')} (카테고리: {seed.get('category')})"
    kor = seed.get("descriptionKor") or ""
    if kor:
        seed_line += f"\n- 시드 설명: {kor[:300]}"
    table = json.dumps(rows, ensure_ascii=False, indent=2)
    return (
        f"{seed_line}\n\n"
        f"[종합 스코어링 상위 아이템 (JSON)]\n{table}\n\n"
        "위 데이터로 다음 구성의 보고서를 작성하세요:\n"
        "## 1. 분석 개요 (시드와 분석 범위)\n"
        "## 2. 유망 아이템 TOP 3 해석 (각 지표가 의미하는 바)\n"
        "## 3. 종합 시사점 및 제언\n"
    )


def _fallback_report(seed: Dict[str, Any], summary: pd.DataFrame) -> str:
    """LLM 미가동 시 데이터로 조립하는 결정적 보고서."""
    name = seed.get("itemName", "-")
    cat = seed.get("category", "-")
    lines = [
        f"# 유망아이템 발굴 보고서 — {cat}",
        "",
        "## 1. 분석 개요",
        f"- **시드 아이템**: {name} (카테고리: {cat})",
    ]
    kor = seed.get("descriptionKor")
    if kor:
        lines.append(f"- **설명**: {str(kor)[:300]}")
    lines += ["", "## 2. 유망 아이템 TOP 3"]
    if summary is not None and not summary.empty:
        top = summary.head(3)
        for _, r in top.iterrows():
            item = r.get("itemName", "-")
            bits = []
            for key, klabel in [("종합점수", "종합"), ("score", "기술집약도"),
                                ("demandEmergence", "수요부상도"), ("supplyEmergence", "공급부상도")]:
                v = r.get(key)
                if pd.notna(v) if v is not None else False:
                    bits.append(f"{klabel} {v}")
            lines.append(f"- **{item}** — " + (", ".join(bits) if bits else "지표 정보 제한적"))
    else:
        lines.append("- (분석 결과가 비어 있습니다)")
    lines += [
        "",
        "## 3. 종합 시사점",
        "- 기술집약도가 높은 아이템은 기술적 성숙·특허 밀집도가 높아 진입장벽과 응용 잠재력이 큽니다.",
        "- 수요/공급 부상도가 함께 상승하는 아이템은 시장·연구가 동시에 확대되는 유망 신호입니다.",
        "- 상위 아이템을 중심으로 세부 기술 로드맵과 연관 아이템 확장 분석을 권장합니다.",
        "",
        "_※ 본 보고서는 LLM 미연결 상태에서 지표 데이터로 자동 조립된 기본 보고서입니다._",
    ]
    return "\n".join(lines)


def generate_report(seed: Dict[str, Any], summary: pd.DataFrame) -> Tuple[str, str]:
    """(보고서 마크다운, 사용 모델명) 반환. LLM 실패 시 폴백.

    로컬 vLLM(OpenAI 호환)을 requests로 직접 호출 → 'openai' 패키지 불필요.
    """
    import requests

    rows = _rows_for_prompt(summary)
    model = config.llm_model()
    # base_url 은 '.../v1' 형태 → chat/completions 엔드포인트 조립
    url = config.llm_base_url().rstrip("/") + "/chat/completions"
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_prompt(seed, rows)},
        ],
        "temperature": 0.4,
        "max_tokens": 1800,
    }
    # Qwen3 계열은 기본 thinking 모드 → 보고서에 추론 텍스트가 섞이지 않도록 비활성화
    if "qwen" in model.lower():
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    try:
        resp = requests.post(
            url,
            json=payload,
            headers={"Authorization": f"Bearer {config.llm_api_key()}"},
            timeout=90,
        )
        resp.raise_for_status()
        content = _clean(resp.json()["choices"][0]["message"]["content"] or "")
        if content:
            return content, model
    except Exception as exc:  # 엔드포인트 미가동/네트워크/응답형식 등
        print(f"[llm] LLM 호출 실패 → 폴백 보고서 사용: {exc}")
    return _fallback_report(seed, summary), "fallback"