# api.py
import asyncio
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

# app2.py의 핵심 로직을 pipeline.py로 분리했다고 가정합니다.
# 2번 항목에서 pipeline.py를 만드는 방법을 설명합니다.
import pipeline

app = FastAPI(
    title="유망 아이템 분석 API",
    description="시드 키워드를 입력받아 위키피디아 네트워크 분석 결과를 반환합니다."
)


# --- API 출력 모델 정의 ---
# Pydantic을 사용하여 API 응답의 구조를 명확하게 정의합니다.

class EntryItem(BaseModel):
    item: str = Field(..., description="위키피디아 실제 제목(True Title)")
    url: Optional[str] = Field(None, description="위키피디아 문서 URL")


class ItemSummary(BaseModel):
    summary: str = Field(..., description="아이템 요약본 (첫 320자)")


class NetworkNode(BaseModel):
    id: str = Field(..., description="노드 이름 (항목 제목)")
    group: int = Field(..., description="커뮤니티 그룹 ID")
    size: float = Field(..., description="노드 크기 (PageRank 기반)")


class NetworkEdge(BaseModel):
    source: str = Field(..., description="시작 노드")
    target: str = Field(..., description="끝 노드")
    weight: float = Field(..., description="엣지 가중치")


class NetworkData(BaseModel):
    nodes: List[NetworkNode]
    edges: List[NetworkEdge]


class TrendData(BaseModel):
    years: List[int] = Field(..., description="연도 목록")
    edits: List[float] = Field(..., description="연도별 편집수")
    views: List[float] = Field(..., description="연도별 조회수")


class AnalysisResult(BaseModel):
    """최종 API 분석 결과 응답 모델"""
    item_1_entry: EntryItem = Field(..., description="1) 진입 아이템")
    item_2_summary: ItemSummary = Field(..., description="2) 아이템 설명")
    item_3_network: NetworkData = Field(..., description="3) 전체 네트워크 (JSON)")
    item_4_indicators: List[Dict[str, Any]] = Field(..., description="4) 유망성 지표 비교 (상위 20개)")
    item_5_trends: TrendData = Field(..., description="5) 연도별 활동 추세")


# --- 멀티 시드 모델 ---

class MultiSeedRequest(BaseModel):
    seeds: List[str] = Field(..., description="분석할 시드 목록 (1~10개)", min_length=1, max_length=10)
    n_depth: int = Field(1, description="네트워크 확장 차수 (1~3)", ge=1, le=3)


class MultiAnalysisResult(BaseModel):
    """멀티 시드 통합 분석 결과 응답 모델"""
    item_1_entries: List[EntryItem] = Field(..., description="1) 시드별 진입 아이템 목록")
    item_2_summaries: List[ItemSummary] = Field(..., description="2) 시드별 아이템 설명 목록")
    item_3_network: NetworkData = Field(..., description="3) 통합 네트워크 (JSON)")
    item_4_indicators: List[Dict[str, Any]] = Field(..., description="4) 통합 유망성 지표 (상위 20개)")
    item_5_trends: Dict[str, TrendData] = Field(..., description="5) 시드별 연도별 활동 추세")
    meta: Dict[str, Any] = Field(..., description="분석 메타정보")


# --- API 엔드포인트 ---

@app.post(
    "/analyze/multi",
    response_model=MultiAnalysisResult,
    summary="멀티 시드 통합 분석",
    description="여러 시드 키워드를 입력받아 개별 분석 후 통합 네트워크 분석 결과를 반환합니다."
)
async def analyze_multi_seed(request: MultiSeedRequest):
    """
    여러 시드를 받아 통합 분석을 수행합니다.

    - **seeds**: 분석할 시드 목록 (예: ["Electric battery", "Solar cell"])
    - **n_depth**: 네트워크 확장 차수 (기본값: 1)
    """
    try:
        print(f"[멀티] API 요청 수신: {request.seeds} (n_depth={request.n_depth})")
        result = await asyncio.to_thread(pipeline.get_multi_analysis_results, request.seeds, request.n_depth)
        print(f"[멀티] 분석 완료. 결과 반환.")
        return result

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"분석 중 필수 파일을 찾을 수 없습니다: {e}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"입력값 오류: {e}")
    except Exception as e:
        print(f"[오류] 멀티 시드: {e}")
        raise HTTPException(status_code=500, detail=f"서버 내부 오류 발생: {e}")


@app.get(
    "/analyze/{seed_name}",
    response_model=AnalysisResult,
    summary="시드 아이템 분석",
    description="시드 이름을 받아 전체 분석 파이프라인을 실행하고 5가지 주요 결과를 JSON으로 반환합니다."
)
async def analyze_seed(seed_name: str, n_depth: int = 1):
    """
    시드 이름을 받아 분석을 수행합니다.

    - **seed_name**: 분석할 시드 이름 (예: "Electric battery")
    - **n_depth**: 네트워크 확장 차수 (기본값: 1)
    """
    try:
        print(f"[{seed_name}] API 요청 수신... (n_depth={n_depth})")

        # pipeline.py에 정의된 메인 함수를 호출합니다.
        # 이 함수는 내부적으로 캐싱을 처리합니다 (파일이 있으면 스킵).
        result = await asyncio.to_thread(pipeline.get_analysis_results, seed_name, n_depth)

        print(f"[{seed_name}] 분석 완료. 결과 반환.")
        return result

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"분석 중 필수 파일을 찾을 수 없습니다: {e}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"입력값 오류: {e}")
    except Exception as e:
        print(f"[오류] {seed_name}: {e}")
        raise HTTPException(status_code=500, detail=f"서버 내부 오류 발생: {e}")


if __name__ == "__main__":
    # API 서버 실행
    # 터미널에서: uvicorn api:app --reload
    print("API 서버를 시작하려면 터미널에서 'uvicorn api:app --reload'를 실행하세요.")
    # uvicorn.run(app, host="0.0.0.0", port=8000)