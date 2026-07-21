# 유망아이템 발굴 실습 — 다른 컴퓨터 세팅 가이드

`edu/` 교육툴(APOLLO 데모 + 위키 직접수집 + 단계별 노트북)을 새 컴퓨터에서 실행하는 절차.

## 0. 사전 확인 (제일 중요)
- **APOLLO 접속**: `apollo.kisti.re.kr` 은 **KISTI 내부망** 주소입니다. 데모 컴퓨터가 **KISTI망(또는 VPN)** 에 있어야 APOLLO 모드가 동작합니다. 외부망이면 데이터 수집이 안 됩니다.
  - 확인: `curl -k https://apollo.kisti.re.kr/service-test/open/api/v1/itemsntop100?category=인공지능&indicator=TECH_INTENSITY`
- **LLM(6단계 보고서)**: vLLM 등 OpenAI 호환 엔드포인트가 있어야 실제 보고서가 나옵니다. 없으면 **지표 기반 폴백 보고서**로 자동 대체(앱은 정상 동작).
- **OS**: Windows 기준. 실행 시 `PYTHONUTF8=1` 필수(cp949 크래시 방지).

## 1. 클론 & 가상환경
```bash
git clone https://github.com/seosumin/wiki_web.git
cd wiki_web
python -m venv venv
./venv/Scripts/pip install --upgrade pip
./venv/Scripts/pip install -r requirements.txt
```

## 2. 설정 (선택) — `.streamlit/secrets.toml`
git으로 전송되지 않으므로(비밀정보) 필요하면 직접 만듭니다. **없어도 동작**합니다(DB=로컬 SQLite 폴백, LLM=기본값).
```toml
[supabase]
# db_url = "postgresql://...:5432/postgres?sslmode=require"   # 없으면 SQLite(edu.db) 사용

[llm]
base_url = "http://localhost:8005/v1"   # 이 컴퓨터의 vLLM 주소로
model = "Qwen/Qwen3-8B"
api_key = "EMPTY"
```
> `.streamlit/config.toml`(CORS/포트 등)은 repo에 포함돼 그대로 옵니다.

## 3. 실행
```bash
# Streamlit 앱
PYTHONUTF8=1 ./venv/Scripts/streamlit.exe run edu/app.py
# 브라우저에서 http://localhost:8501
```

## 4. 공개 링크가 필요하면 (원격 교육생)
- Cloudflare 터널: `edu/run_tunnel.sh` 참고. cloudflared는 **WSL에서** 실행(KISTI가 Windows TLS 가로챔).
  - `bash edu/run_tunnel.sh 8501` → trycloudflare 임시 URL 공유.

## 5. 단계별 노트북 (`edu/notebooks/`)
설명+코드+결과를 담은 학습용 노트북 00~06. 실행하려면 Jupyter 커널 필요:
```bash
./venv/Scripts/pip install jupyterlab ipykernel   # requirements에 포함됨
PYTHONUTF8=1 ./venv/Scripts/jupyter-lab.exe        # 또는 VSCode에서 venv 커널 선택
```
- `06_AI_보고서.ipynb` 는 vLLM 서버가 켜져 있어야 동작.

## 전송되지 않는 것 (`.gitignore`)
`venv/`, `.streamlit/secrets.toml`, `runs/`(크롤 캐시), `edu.db`, vLLM 모델·cloudflared 바이너리 → 위 절차대로 각자 준비.

## 데이터 참고
- 테스트 서버(`service-test`)는 일부 카테고리만 데이터 제공(양자·인공지능·반도체·우주항공·첨단바이오). 운영 서버 전환 시 `edu/apollo.py`의 `BASE_URL` 변경.
- 위키 직접수집은 1차수 고정, 아이템당 수 분 소요(XTools). 동시 1건 직렬화 락 적용.
