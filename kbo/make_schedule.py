"""
잔여 경기 일정 생성기 (임시)
=============================

⚠️ 이 파일은 **임시 대체물(placeholder)** 이다.
   실제로는 KBO 공식 홈페이지의 잔여 경기 일정을 받아 data/schedule.csv를
   교체해야 한다. 여기서는 순위표의 '팀별 잔여 경기 수'만 만족하도록
   대진을 균등 배분해 형태가 같은 파일을 만든다.

   → 확률의 절대값은 실제 일정으로 교체한 뒤에야 의미가 있다.
     지금은 파이프라인 검증용으로만 쓴다.
"""

import numpy as np
import pandas as pd
from datetime import date, timedelta

from kbo_sim import TOTAL_GAMES

SEASON_END = date(2026, 10, 3)
TODAY = date(2026, 8, 21)


def build(standings: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    teams = standings["team"].tolist()
    played = (standings["w"] + standings["d"] + standings["l"]).to_numpy()
    remain = (TOTAL_GAMES - played).astype(int)

    assert remain.sum() % 2 == 0, "팀별 잔여 경기 합이 홀수 — 순위표를 확인하세요"

    # 잔여 경기가 가장 많은 팀을 고르고, 상대는 '잔여 경기 수에 비례한 확률'로 추출한다.
    #   - argmax로 고정하면 끝에 남은 두 팀끼리 30경기가 잡히는 쏠림이 생긴다.
    #   - 확률 추출로 바꾸면 매치업이 고르게 퍼지면서도 팀별 잔여 경기 수는 정확히 지켜진다.
    # 매 단계에서 max(r) <= sum(r) - max(r) 이면 교착에 빠지지 않는다.
    pairs = []
    r = remain.astype(float).copy()
    while r.sum() > 0:
        i = int(np.argmax(r))
        w = r.copy()
        w[i] = 0
        if w.sum() == 0:
            raise RuntimeError("대진 배정 교착 — 잔여 경기 분포를 확인하세요")
        j = int(rng.choice(len(w), p=w / w.sum()))
        pairs.append((i, j))
        r[i] -= 1
        r[j] -= 1

    # 홈/원정 균등 배분 + 날짜 배정
    days = (SEASON_END - TODAY).days
    rows = []
    for n, (i, j) in enumerate(pairs):
        if rng.random() < 0.5:
            i, j = j, i
        rows.append({
            "date": (TODAY + timedelta(days=int(n / len(pairs) * days) + 1)).isoformat(),
            "home": teams[i],
            "away": teams[j],
        })

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


if __name__ == "__main__":
    st = pd.read_csv("data/standings.csv")
    sch = build(st)
    sch.to_csv("data/schedule.csv", index=False)
    print(f"잔여 경기 {len(sch)}경기 생성 → data/schedule.csv")
    print(sch.head())
