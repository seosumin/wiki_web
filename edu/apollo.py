# -*- coding: utf-8 -*-
"""
APOLLO Open API (A4: 글로벌 유망 아이템 탐색 서비스) 클라이언트.

교육용 파이프라인의 '데이터 수집' 계층. 기존 위키피디아/XTools 크롤링을
KISTI 공식 API 호출로 대체한다.

주의:
- 테스트 서버는 사설 인증서라 verify=False 로 호출한다 (로컬/KISTI망 실행 전제).
- 네트워크(A4.4/A4.5)는 degree=2, indicator=true 조합에서 데이터가 나온다.
- 아이템에 따라 네트워크가 비어있거나 일시적으로 'AI API error'가 날 수 있어 재시도한다.

참고: https://apollo.kisti.re.kr/service-test/swagger-ui/index.html
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BASE_URL = "https://apollo.kisti.re.kr/service-test"
HEADERS = {"Content-Type": "application/json"}
TIMEOUT = 120

# 국가전략기술 카테고리 (A4.1 TOP100 조회용, 데모 프리셋)
CATEGORIES = [
    "양자", "인공지능", "반도체·디스플레이", "이차전지",
    "우주항공·해양", "차세대원자력", "수소", "사이버보안",
    "첨단바이오", "차세대통신", "첨단로봇·제조", "첨단모빌리티",
]

# A4.1 지표
INDICATORS = {
    "SUPPLY_EMERGENCE": "공급 부상도",
    "DEMAND_EMERGENCE": "수요 부상도",
    "TECH_INTENSITY": "기술 집약도",
}


class ApolloError(RuntimeError):
    """APOLLO API 호출 실패."""


def _request(method: str, path: str, *, retries: int = 3, backoff: float = 2.0,
             **kwargs) -> requests.Response:
    """5xx/일시오류(AI API error) 시 재시도하는 요청 헬퍼."""
    url = f"{BASE_URL}{path}"
    last_exc: Optional[Exception] = None
    resp: Optional[requests.Response] = None
    for attempt in range(retries + 1):
        try:
            resp = requests.request(
                method, url, headers=HEADERS, verify=False, timeout=TIMEOUT, **kwargs
            )
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(backoff)
                continue
            raise ApolloError(f"요청 실패 {method} {path}: {exc}") from exc

        # 게이트웨이 5xx → 재시도
        if resp.status_code >= 500 and attempt < retries:
            time.sleep(backoff)
            continue
        # AI 연산 일시 오류(E00011) → 재시도
        if resp.status_code == 400 and attempt < retries:
            try:
                code = resp.json().get("code", "")
            except Exception:
                code = ""
            if code == "E00011":
                time.sleep(backoff)
                continue
        break
    if resp is None:
        raise ApolloError(f"요청 실패 {method} {path}: {last_exc}")
    return resp


def _data(resp: requests.Response) -> Any:
    """meta/data 래퍼를 벗겨 data 페이로드를 반환."""
    try:
        body = resp.json()
    except ValueError as exc:
        raise ApolloError(f"JSON 파싱 실패 (HTTP {resp.status_code}): {resp.text[:200]}") from exc

    # 정상 응답: {"meta": {"code":200,...}, "data": ...}
    if isinstance(body, dict) and "data" in body:
        return body["data"]
    # 오류 응답: {"code":"E...", "message":"..."}
    msg = body.get("message", "") if isinstance(body, dict) else ""
    code = body.get("code", "") if isinstance(body, dict) else ""
    raise ApolloError(f"API 오류 (HTTP {resp.status_code}) {code}: {msg}")


# ============================================================
# A4.2 글로벌 아이템 검색  →  시드 아이템 확정 (기존 '1단계 시드 확인' 대체)
# ============================================================
def search_items(query: str, search_type: str = "NAME") -> List[Dict[str, Any]]:
    """
    아이템 검색. 반환 각 원소:
      {rank, itemId, itemName, category,
       scores:{techIntensity, demandEmergence, supplyEmergence},
       description:{en, ko}}
    """
    resp = _request("POST", "/open/api/v1/items/search",
                    json={"searchType": search_type, "query": query})
    data = _data(resp)
    return data.get("items", []) if isinstance(data, dict) else []


# ============================================================
# A4.1 국가전략기술 카테고리별 TOP 100
# ============================================================
def top100(category: str, indicator: str = "SUPPLY_EMERGENCE") -> List[Dict[str, Any]]:
    """
    카테고리+지표별 TOP100. 반환 각 원소:
      {rank, itemId, itemName, score, description, descriptionKor}
    """
    resp = _request("GET", "/open/api/v1/itemsntop100",
                    params={"category": category, "indicator": indicator})
    data = _data(resp)
    return data.get("items", []) if isinstance(data, dict) else []


# ============================================================
# A4.3 아이템 상세  →  지표/연도별 트렌드 (기존 'XTools 수집' 대체)
# ============================================================
def item_details(item_id: int) -> Dict[str, Any]:
    """
    아이템 상세. 반환:
      {itemId, itemName, category, description:{en,ko},
       currentScores:{demandEmergence, supplyEmergence},
       yearlyTrends:[{year, demand, supply}, ...]}
    """
    resp = _request("GET", f"/open/api/v1/items/{item_id}/details")
    data = _data(resp)
    return data if isinstance(data, dict) else {}


# ============================================================
# A4.4 아이템 네트워크  →  {nodes, edges} 그래프
# ============================================================
def item_network(item_id: int, degree: int = 2, indicator: bool = True) -> Dict[str, List]:
    """네트워크 그래프. 반환 {'nodes': [...], 'edges': [...]}. 비어있을 수 있음."""
    resp = _request("GET", f"/open/api/v1/items/{item_id}/network",
                    params={"degree": degree, "indicator": str(indicator).lower()})
    try:
        data = _data(resp)
    except ApolloError:
        return {"nodes": [], "edges": []}
    if not isinstance(data, dict):
        return {"nodes": [], "edges": []}
    return {"nodes": data.get("nodes", []) or [], "edges": data.get("edges", []) or []}


# ============================================================
# A4.5 네트워크 연관 아이템 리스트  →  지표 포함 연관 아이템
# ============================================================
def network_list(item_id: int, degree: int = 2) -> List[Dict[str, Any]]:
    """
    연관 아이템 리스트 (지표 포함). 반환 각 원소:
      {rank, itemName, category, techIntensity, demandEmergence, supplyEmergence}
    A4.4보다 안정적이고 지표까지 있어 네트워크 구성의 주 소스로 사용.
    """
    resp = _request("GET", f"/open/api/v1/items/{item_id}/network/list",
                    params={"degree": degree})
    try:
        data = _data(resp)
    except ApolloError:
        return []
    return data if isinstance(data, list) else []


def health_check() -> bool:
    """API 도달 가능 여부 간단 확인."""
    try:
        top100(CATEGORIES[0], "SUPPLY_EMERGENCE")
        return True
    except Exception:
        return False
