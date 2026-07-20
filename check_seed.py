# check_seed.py
import os
import pandas as pd
import urllib3
from typing import List, Union

import wiki_crawling  # ← 기존 모듈 그대로 사용

urllib3.disable_warnings()

def _normalize_seeds(obj: Union[str, pd.DataFrame, List[str]]) -> List[str]:
    """
    - 문자열 1개, 리스트[str], DataFrame(Title/True_title/seed 유사 컬럼) 모두 지원
    - 공백/양끝 공백 제거, 빈 값 제거, 중복 제거
    """
    seeds: List[str] = []

    if isinstance(obj, str):
        seeds = [obj]
    elif isinstance(obj, list):
        seeds = [str(x) for x in obj]
    elif isinstance(obj, pd.DataFrame):
        # 우선순위: True_title > Title > 첫 번째 문자열 컬럼
        cand_cols = [c for c in obj.columns if str(c).lower() in ("true_title", "title", "seed", "name")]
        if not cand_cols:
            # 문자열 dtype 컬럼만 취합 (최초 1개 사용)
            text_cols = [c for c in obj.columns if pd.api.types.is_string_dtype(obj[c])]
            if not text_cols:
                raise ValueError("DataFrame에서 seed로 사용할 문자열 컬럼을 찾을 수 없습니다.")
            use_col = text_cols[0]
        else:
            use_col = cand_cols[0]
        seeds = obj[use_col].astype(str).tolist()
    else:
        raise TypeError("지원하지 않는 입력 타입입니다. str, list[str], DataFrame 중 하나를 사용하세요.")

    # 정규화: strip → 빈문자 제거 → 중복 제거 → 원래 순서 보존
    seeds = [s.strip() for s in seeds if isinstance(s, str)]
    seeds = [s for s in seeds if s]  # 빈값 제거
    # 순서 보존 중복 제거
    seen = set()
    deduped = []
    for s in seeds:
        if s not in seen:
            seen.add(s)
            deduped.append(s)
    return deduped


def resolve_true_titles(
    seed_input: Union[str, List[str], pd.DataFrame],
) -> pd.DataFrame:
    """
    wiki_crawling.check_seed(dataset) 호출해 결과 DataFrame 반환.
    반환 컬럼: ['No', 'Title', 'True_title','Category','Contents', 'URL']
    """
    dataset = _normalize_seeds(seed_input)
    if not dataset:
        raise ValueError("입력된 seed가 없습니다.")

    row_list = wiki_crawling.check_seed(dataset)
    df = pd.DataFrame(row_list, columns=['No', 'Title', 'True_title', 'Category', 'Contents', 'URL'])

    # 후처리(선택): 공백/언더스코어 통일 필요시 아래 중 택1
    # df['Title'] = df['Title'].str.replace('_', ' ', regex=False)
    # df['True_title'] = df['True_title'].str.replace('_', ' ', regex=False)

    return df


def save_true_seeds(df: pd.DataFrame, out_dir: str = "./data/", out_name: str = "True_seed.xlsx") -> str:
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, out_name)
    # 보기 좋게 인덱스 제거 + 제목 정렬
    df = df.sort_values('True_title').reset_index(drop=True)
    df.to_excel(out_path, index=False)
    return out_path


# CLI로도 쓸 수 있게 (선택)
if __name__ == "__main__":
    data_path = "./data/"
    data_name = "seed.xlsx"

    # Excel에서 읽고 싶으면:
    seed_df = pd.read_excel(os.path.join(data_path, data_name))
    # 단일 seed로 테스트하려면:
    # seed_df = "Nvidia"

    result = resolve_true_titles(seed_df)
    path = save_true_seeds(result, out_dir=data_path, out_name="True_seed.xlsx")
    print("Saved:", path)