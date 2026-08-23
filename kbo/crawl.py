"""
KBO 순위 수집기
================

네이버 스포츠 내부 API에서 순위를 받아 data/standings.csv를 갱신한다.

HTML 파싱 대신 API를 직접 호출하는 이유:
네이버·다음·KBO 공식 모두 순위표를 JavaScript로 렌더링하므로,
requests로 받은 HTML에는 표 껍데기만 있고 숫자가 없다.
에러 없이 빈 표를 반환하기 때문에 조용히 실패한다.

수집 데이터는 반드시 검증한 뒤에만 저장한다. 크롤링의 실패 방식은
예외가 아니라 '그럴듯한 빈 값'이므로, 저장 전 검사가 유일한 방어선이다.
"""

import sys
import pandas as pd
import requests

API = ("https://api-gw.sports.naver.com/statistics/categories/kbo"
       "/seasons/2026/teams?gameType=REGULAR_SEASON")

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    # 네이버 API는 네이버 페이지에서 온 요청만 받는 경우가 있다
    "Referer": "https://m.sports.naver.com/kbaseball/record/kbo",
}

TOTAL_GAMES = 144
N_TEAMS = 10
OUT = "data/standings.csv"


def fetch() -> pd.DataFrame:
    r = requests.get(API, headers=HEADERS, timeout=20)
    r.raise_for_status()
    body = r.json()

    if not body.get("success"):
        raise RuntimeError(f"API가 실패를 반환: code={body.get('code')}")

    rows = [{
        "team": t["teamName"],
        "w": t["winGameCount"],
        "d": t["drawnGameCount"],
        "l": t["loseGameCount"],
        "g": t["gameCount"],
    } for t in body["result"]["seasonTeamStats"]]

    return pd.DataFrame(rows)


def validate(df: pd.DataFrame):
    """저장 전 검사. 하나라도 실패하면 기존 파일을 건드리지 않는다."""

    # [1] 팀 수 — 빈 응답이나 부분 응답을 잡는다
    if len(df) != N_TEAMS:
        raise ValueError(f"팀 수가 {len(df)}개 (10개여야 함)")

    # [2] 승·무·패 합 = 경기 수 — 필드를 잘못 매핑했는지 잡는다
    bad = df[df.w + df.d + df.l != df.g]
    if len(bad):
        raise ValueError(f"승무패 합이 경기 수와 불일치:\n{bad}")

    # [3] 잔여 경기 합은 짝수 — 경기는 두 팀이 함께 치른다
    remain = (TOTAL_GAMES - df.g).sum()
    if remain % 2 != 0:
        raise ValueError(f"잔여 경기 합이 홀수({remain})")
    if remain < 0:
        raise ValueError(f"잔여 경기가 음수({remain}) — 총 경기 수 확인 필요")

    # [4] 경기 수는 줄어들 수 없다 (단조 증가)
    #     주의: KBO는 월요일이 휴식일이므로 '어제와 같음'은 정상이다.
    #     같은 값을 오류로 처리하면 매주 화요일마다 오탐이 발생한다.
    try:
        prev = pd.read_csv(OUT)
        prev_total = (prev.w + prev.d + prev.l).sum()
        now_total = df.g.sum()
        if now_total < prev_total:
            raise ValueError(f"총 경기 수가 감소({prev_total} → {now_total})")
        if now_total == prev_total:
            print("  [INFO] 총 경기 수 변동 없음 (휴식일이거나 전 경기 취소)")
    except FileNotFoundError:
        pass


if __name__ == "__main__":
    try:
        df = fetch()
        validate(df)
    except Exception as e:
        # 검증 실패 시 기존 파일을 보존한다. 낡은 데이터가
        # 잘못된 데이터보다 낫고, 정합성 검사가 뒤에서 한 번 더 잡는다.
        print(f"  [FAIL] 수집 중단 — {type(e).__name__}: {e}")
        sys.exit(1)

    df[["team", "w", "d", "l"]].to_csv(OUT, index=False)
    print(f"  [OK] {len(df)}개 구단 저장 → {OUT}")
    print(f"       총 {df.g.sum()}경기 소화 · 잔여 {(TOTAL_GAMES - df.g).sum()}경기")
    print(df.sort_values("w", ascending=False).to_string(index=False))
