#!/usr/bin/env bash
# =============================================================
# 교육툴 공개 링크용 Cloudflare 터널 (WSL 전용).
# KISTI 망이 Windows TLS를 가로채므로 cloudflared는 반드시 WSL에서 실행.
#
# 사용법 (WSL 터미널에서):
#   bash /mnt/f/wiki_web/edu/run_tunnel.sh          # 기본 8501
#   bash /mnt/f/wiki_web/edu/run_tunnel.sh 8501
#
# 사전조건: Windows에서 Streamlit이 0.0.0.0:<PORT> 로 실행 중이어야 함
#   PYTHONUTF8=1 ./venv/Scripts/streamlit.exe run edu/app.py
# =============================================================
set -euo pipefail

PORT="${1:-8501}"
CF="${CLOUDFLARED:-/home/sumin/cloudflared}"

# WSL에서 본 Windows 호스트 IP (기본 게이트웨이). 재부팅 시 바뀔 수 있어 매번 조회.
WINHOST="$(ip route show default | awk '{print $3}')"

echo "[tunnel] Windows host = $WINHOST, port = $PORT"
echo "[tunnel] 로컬 도달 확인..."
if ! curl -s -o /dev/null --max-time 8 "http://$WINHOST:$PORT"; then
  echo "[tunnel] ERROR: http://$WINHOST:$PORT 에 연결 실패. Streamlit이 0.0.0.0:$PORT 로 떠 있는지 확인하세요." >&2
  exit 1
fi

echo "[tunnel] cloudflared 시작 (http2)…  아래 trycloudflare.com URL을 교육생에게 공유하세요."
exec "$CF" tunnel --protocol http2 --url "http://$WINHOST:$PORT"
