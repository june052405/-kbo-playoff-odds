"""매일 실행되는 메인 스크립트."""

import pandas as pd
from kbo_sim import simulate, validate, leverage

TARGET = "롯데"
N_SIMS = 200_000

standings = pd.read_csv("data/standings.csv")
schedule = pd.read_csv("data/schedule.csv")

prob, made, outcome, teams = simulate(standings, schedule, n_sims=N_SIMS)

print("=" * 52)
print(f"  가을야구 진출 확률  (잔여 {len(schedule)}경기 · {N_SIMS:,}회 시뮬레이션)")
print("=" * 52)
res = (pd.DataFrame({"team": teams, "prob": prob})
       .sort_values("prob", ascending=False))
for _, r in res.iterrows():
    bar = "█" * int(r.prob * 30)
    mark = " ←" if r.team == TARGET else ""
    print(f"  {r.team:<5} {r.prob*100:6.2f}%  {bar}{mark}")

print("\n[검증]")
for m in validate(prob, N_SIMS, schedule, standings):
    print("  " + m)

lev = leverage(made, outcome, schedule, teams, TARGET)
print(f"\n[{TARGET} 잔여 경기 영향도 TOP 5]")
print("  승패에 따라 진출 확률이 가장 크게 갈리는 경기")
for _, r in lev.head(5).iterrows():
    print(f"  {r.date}  vs {r.opponent:<4}({r.venue})  "
          f"승={r.p_if_win*100:5.2f}%  패={r.p_if_lose*100:5.2f}%  "
          f"영향도={r.leverage*100:+.2f}%p")

# --- 기록 · 브리핑 · 알림 ---------------------------------------------------
import notify

p = float(prob[teams.index(TARGET)])
rank = int(res.reset_index(drop=True).query("team == @TARGET").index[0]) + 1

hist = notify.record(p, rank)
notify.plot(hist)
delta = p - hist.prob.iloc[-2] if len(hist) > 1 else 0.0

top = lev.iloc[0].to_dict()
text = notify.brief(p, delta, rank, top)

print("\n[브리핑]")
print("  " + text.replace("\n", "\n  "))
notify.send(text, "output/trend.png")
