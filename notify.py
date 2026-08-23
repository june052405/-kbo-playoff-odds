"""
결과 기록 · 브리핑 생성 · 알림 전송
====================================

생성형 AI(Claude API)를 파이프라인의 '구성요소'로 사용한다.
숫자 계산은 전적으로 시뮬레이션이 담당하고, LLM은 이미 확정된 수치를
읽기 좋은 한 문단으로 옮기는 역할만 맡는다.
LLM에게 확률을 추정시키지 않는 이유는 검증이 불가능하기 때문이다.
"""

import os
import json
import urllib.request
from datetime import datetime
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HISTORY = "data/history.csv"
CHART = "output/trend.png"
TARGET = "롯데"


# --- 1. 이력 누적 -----------------------------------------------------------
def record(prob: float, rank: int) -> pd.DataFrame:
    today = datetime.now(ZoneInfo("Asia/Seoul")).date().isoformat()
    row = pd.DataFrame([{"date": today, "prob": prob, "rank": rank}])
    if os.path.exists(HISTORY):
        hist = pd.read_csv(HISTORY)
        hist = hist[hist.date != row.date.iloc[0]]          # 같은 날 재실행 시 갱신
        hist = pd.concat([hist, row], ignore_index=True)
    else:
        hist = row
    hist.sort_values("date").to_csv(HISTORY, index=False)
    return hist


# --- 2. 추이 그래프 ---------------------------------------------------------
def plot(hist: pd.DataFrame):
    os.makedirs("output", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(pd.to_datetime(hist.date), hist.prob * 100,
            marker="o", markersize=4, linewidth=1.8, color="#C60C30")
    ax.fill_between(pd.to_datetime(hist.date), hist.prob * 100,
                    alpha=0.15, color="#C60C30")
    ax.set_ylabel("Playoff probability (%)")
    # 차트는 영문만 사용한다. CI 러너에 한글 폰트가 없어 로컬에서만 정상 렌더링되는
    # 문제를 피하기 위함 (로컬 통과 → CI 실패의 전형적인 원인).
    ax.set_title("Lotte Giants — daily playoff odds", loc="left")
    ax.grid(alpha=0.25, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(CHART, dpi=140)
    plt.close(fig)


# --- 3. 브리핑 생성 (Claude API) --------------------------------------------
def brief(prob: float, delta: float, rank: int, top_game: dict) -> str:
    """확정된 수치를 한 문단 브리핑으로 변환. 실패해도 파이프라인은 계속 동작한다."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    fallback = (f"{TARGET} 가을야구 진출 확률 {prob*100:.2f}% "
                f"({delta*100:+.2f}%p) · 현재 {rank}번째")
    if not key:
        return fallback

    payload = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 300,
        "system": ("너는 야구 데이터 브리핑을 쓴다. 주어진 수치만 사용하고 "
                   "새로운 숫자를 만들어내지 마라. 3문장 이내, 담백한 존댓말."),
        "messages": [{"role": "user", "content": json.dumps({
            "팀": TARGET,
            "진출확률": f"{prob*100:.2f}%",
            "전일대비": f"{delta*100:+.2f}%p",
            "확률순위": rank,
            "최대분수령": f"{top_game['date']} vs {top_game['opponent']} "
                          f"(영향도 {top_game['leverage']*100:+.2f}%p)",
        }, ensure_ascii=False)}],
    }
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(payload).encode(),
        headers={"content-type": "application/json",
                 "x-api-key": key,
                 "anthropic-version": "2023-06-01"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            body = json.load(r)
        return "".join(b.get("text", "") for b in body["content"]).strip()
    except Exception as e:
        print(f"  [WARN] 브리핑 생성 실패 ({e}) — 기본 문구 사용")
        return fallback


# --- 4. Discord 전송 --------------------------------------------------------
def send(text: str, chart_path: str | None = None):
    url = os.environ.get("DISCORD_WEBHOOK")
    if not url:
        print("  [SKIP] DISCORD_WEBHOOK 미설정 — 콘솔 출력으로 대체")
        print(text)
        return

    boundary = "----kbo"
    parts = [f"--{boundary}\r\n"
             f'Content-Disposition: form-data; name="payload_json"\r\n\r\n'
             f'{json.dumps({"content": text}, ensure_ascii=False)}\r\n'.encode()]
    if chart_path and os.path.exists(chart_path):
        parts.append(
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="files[0]"; '
            f'filename="trend.png"\r\nContent-Type: image/png\r\n\r\n'.encode()
            + open(chart_path, "rb").read() + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())

    req = urllib.request.Request(
        url, data=b"".join(parts),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        print(f"  [OK] 알림 전송 완료 (HTTP {r.status})")
