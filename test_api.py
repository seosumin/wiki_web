# -*- coding: utf-8 -*-
"""
APOLLO Open API 테스트 스크립트 (최신 스펙 반영: 2026-06-15)
- A1: 유망 사업화 국가R&D 예측 서비스
- A2: 유망 사업화 수요 기업 예측 서비스
- A4: 글로벌 유망 아이템 탐색 서비스

참고: https://apollo.kisti.re.kr/service-test/swagger-ui/index.html
"""

import sys
import io
import time
import requests
import json
import urllib3
from datetime import datetime

# Windows 콘솔 UTF-8 출력
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# SSL 경고 비활성화 (테스트 환경)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ============================================================
# 설정
# ============================================================
BASE_URL = "https://apollo.kisti.re.kr/service-test"
HEADERS = {"Content-Type": "application/json"}


def request_api(method, path, retries=2, **kwargs):
    """5xx(게이트웨이 일시오류) 시 재시도하는 요청 헬퍼"""
    url = f"{BASE_URL}{path}"
    resp = None
    for attempt in range(retries + 1):
        resp = requests.request(method, url, headers=HEADERS, verify=False, timeout=120, **kwargs)
        if resp.status_code < 500:
            break
        if attempt < retries:
            time.sleep(2)
    return resp


def print_result(test_name, response):
    """테스트 결과 출력 헬퍼"""
    print(f"\n{'='*70}")
    print(f"[테스트] {test_name}")
    print(f"{'='*70}")
    print(f"  URL:        {response.request.method} {response.url}")
    print(f"  Status:     {response.status_code}")
    try:
        body = response.json()
        print(f"  Response:   {json.dumps(body, ensure_ascii=False, indent=4)[:2000]}")
    except Exception:
        print(f"  Response:   {response.text[:1000]}")
    status = "PASS" if response.status_code == 200 else "FAIL"
    print(f"  결과:       [{status}]")
    return response.status_code == 200


# ============================================================
# A1: 유망 사업화 국가R&D 예측 서비스
# ============================================================
# A1.1 의 응답에서 가져온 과제ID 를 이후 조회에 재사용
PROJECT_ID = "1415090620"


def test_a1_1_predictions_rnd():
    """A1.1 유망 R&D Top N 예측 (재무 단위: 백만원)"""
    global PROJECT_ID
    payload = {
        "topN": 100,
        "companyName": "수민전자",
        "establishmentDate": "2020-01-01",
        "location": "대전광역시",
        "industryCode": "C26229",
        "mainProducts": ["진공청소기"],
        "bizPurpose": ["R&D"],
        "revenue": 5000,        # 단위: 백만원
        "totalCapital": 2000,   # 단위: 백만원
        "totalAssets": 7000,    # 단위: 백만원
        "bizRegNumber": "123-45-67890",
    }
    resp = request_api("POST", "/open/api/v1/predictions/RnD", json=payload)
    try:
        PROJECT_ID = resp.json()["data"][0]["projectId"]
    except Exception:
        pass
    return print_result("A1.1 유망 R&D Top N 예측", resp)


def test_a1_2_project():
    """A1.2 과제 조회 (신규)"""
    resp = request_api("GET", f"/open/api/v1/projects/{PROJECT_ID}")
    return print_result("A1.2 과제 조회", resp)


def test_a1_3_project_comprehensive():
    """A1.3 과제 종합조회 (신규)"""
    resp = request_api("GET", f"/open/api/v1/projects/{PROJECT_ID}/comprehensive")
    return print_result("A1.3 과제 종합조회", resp)


def test_a1_4_project_content():
    """A1.4 연구 상세 내용 및 목표"""
    resp = request_api("GET", f"/open/api/v1/projects/{PROJECT_ID}/content")
    return print_result("A1.4 연구 상세 내용 및 목표", resp)


def test_a1_5_project_outcomes():
    """A1.5 연구개발 성과 조회"""
    resp = request_api("GET", f"/open/api/v1/projects/{PROJECT_ID}/outcomes")
    return print_result("A1.5 연구개발 성과 조회", resp)


# ============================================================
# A2: 유망 사업화 수요 기업 예측 서비스
# ============================================================
# 검증된 예시 사업자번호 (기업/종합/현황 조회용)
BUSINESS_NO = "3148642941"


def test_a2_1_predictions_demand_companies():
    """A2.1 유망 수요 기업 Top N 예측 (budget 단위: 백만원, preCompletion 신규)"""
    payload = {
        "topN": 100,
        "projectName": "고효율 나노코팅 공정기술 개발",
        "researchSubjectType": "중소기업",
        "techClassLarge": "재료",
        "techClassMedium": "고분자재료",
        "researchStage": "개발연구",
        "completionYear": "2025",
        "preCompletion": False,   # 신규 선택 필드
        "region": "대전광역시",
        "budget": 1684,           # 단위: 백만원
        "researcherCount": 8,
        "keywords": ["나노코팅", "기능성 필름"],
    }
    resp = request_api("POST", "/open/api/v1/predictions/demand-companies", json=payload)
    return print_result("A2.1 유망 수요 기업 Top N 예측", resp)


def test_a2_2_company():
    """A2.2 기업 조회 (신규)"""
    resp = request_api("GET", f"/open/api/v1/companies/{BUSINESS_NO}")
    return print_result("A2.2 기업 조회", resp)


def test_a2_3_company_comprehensive():
    """A2.3 기업 종합조회 (신규)"""
    resp = request_api("GET", f"/open/api/v1/companies/{BUSINESS_NO}/comprehensive")
    return print_result("A2.3 기업 종합조회", resp)


def test_a2_4_company_status():
    """A2.4 기업 현황 및 비교 통계"""
    resp = request_api("GET", f"/open/api/v1/companies/{BUSINESS_NO}/status")
    return print_result("A2.4 기업 현황 및 비교 통계", resp)


# ============================================================
# A4: 글로벌 유망 아이템 탐색 서비스
# ============================================================
ITEM_ID = 31859187  # itemId 는 정수형


def test_a4_1_items_top100():
    """A4.1 국가전략기술 카테고리별 TOP 100 조회"""
    params = {"category": "양자", "indicator": "SUPPLY_EMERGENCE"}
    resp = request_api("GET", "/open/api/v1/itemsntop100", params=params)
    return print_result("A4.1 국가전략기술 TOP 100 조회", resp)


def test_a4_2_items_search():
    """A4.2 글로벌 아이템 검색"""
    payload = {"searchType": "NAME", "query": "Bay Bridge Series"}
    resp = request_api("POST", "/open/api/v1/items/search", json=payload)
    return print_result("A4.2 글로벌 아이템 검색", resp)


def test_a4_3_item_details():
    """A4.3 아이템 상세 조회"""
    resp = request_api("GET", f"/open/api/v1/items/{ITEM_ID}/details")
    return print_result("A4.3 아이템 상세 조회", resp)


def test_a4_4_item_network():
    """A4.4 아이템 네트워크 조회"""
    params = {"degree": 2, "indicator": "true"}
    resp = request_api("GET", f"/open/api/v1/items/{ITEM_ID}/network", params=params)
    return print_result("A4.4 아이템 네트워크 조회", resp)


def test_a4_5_item_network_list():
    """A4.5 네트워크 연관 아이템 리스트"""
    params = {"degree": 2}
    resp = request_api("GET", f"/open/api/v1/items/{ITEM_ID}/network/list", params=params)
    return print_result("A4.5 네트워크 연관 아이템 리스트", resp)


# ============================================================
# 메인 실행
# ============================================================
if __name__ == "__main__":
    print(f"\n APOLLO Open API 테스트 시작 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Base URL: {BASE_URL}")
    print(f"{'='*70}")

    tests = [
        # A1
        ("A1.1", test_a1_1_predictions_rnd),
        ("A1.2", test_a1_2_project),
        ("A1.3", test_a1_3_project_comprehensive),
        ("A1.4", test_a1_4_project_content),
        ("A1.5", test_a1_5_project_outcomes),
        # A2
        ("A2.1", test_a2_1_predictions_demand_companies),
        ("A2.2", test_a2_2_company),
        ("A2.3", test_a2_3_company_comprehensive),
        ("A2.4", test_a2_4_company_status),
        # A4
        ("A4.1", test_a4_1_items_top100),
        ("A4.2", test_a4_2_items_search),
        ("A4.3", test_a4_3_item_details),
        ("A4.4", test_a4_4_item_network),
        ("A4.5", test_a4_5_item_network_list),
    ]

    results = {}
    for name, test_fn in tests:
        try:
            results[name] = test_fn()
        except Exception as e:
            print(f"\n{'='*70}")
            print(f"[테스트] {name} - 예외 발생: {e}")
            results[name] = False

    # 요약
    print(f"\n\n{'='*70}")
    print(" 테스트 결과 요약")
    print(f"{'='*70}")
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    for name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"  {name:10} [{status}]")
    print(f"\n  총 {total}건 중 {passed}건 성공, {total - passed}건 실패")
    print(f"{'='*70}\n")
