"""
검증 절차가 잡아낸 두 가지 결함 재현
=====================================

초기 구현에서 실제로 발견하고 수정한 결함을 재현한다.
둘 다 코드는 정상 실행되고 그럴듯한 숫자를 뱉는다 —
검증 불변식이 없었다면 그대로 믿었을 종류의 오류다.
"""

import numpy as np
import pandas as pd
from kbo_sim import log5, game_probabilities, PLAYOFF_SPOTS, HFA_ODDS, P_TIE

standings = pd.read_csv("data/standings.csv")
teams = standings["team"].tolist()
talent = (standings.w / (standings.w + standings.l)).to_numpy()

print("=" * 64)
print("  결함 1 — 상대 전력을 무시한 승률 모델")
print("=" * 64)
print("""
  초기 구현: P(홈팀 승) = 홈팀의 시즌 승률
  얼핏 타당해 보이지만, 한 경기의 승/패 확률 합이 1이 되지 않는다.
""")

i, j = teams.index("KT"), teams.index("키움")
print(f"  [{teams[i]} vs {teams[j]}]  승률 {talent[i]:.3f} vs {talent[j]:.3f}")
print(f"    단순 모델 : P(홈승)={talent[i]:.3f}  P(원정승)={talent[j]:.3f}  "
      f"합={talent[i]+talent[j]:.3f}  ← 1이 아님")

ph, pa = game_probabilities(talent[i:i+1], talent[j:j+1])
print(f"    log5 모델 : P(홈승)={ph[0]:.3f}  P(원정승)={pa[0]:.3f}  "
      f"P(무)={P_TIE:.3f}  합={ph[0]+pa[0]+P_TIE:.3f}  ← 정상")

i, j = teams.index("한화"), teams.index("롯데")
print(f"\n  [{teams[i]} vs {teams[j]}]  승률 {talent[i]:.3f} vs {talent[j]:.3f}")
print(f"    단순 모델 : 합={talent[i]+talent[j]:.3f}  ← 이번엔 1보다 작음")
print("""
  → 오차의 부호가 매치업마다 뒤집힌다. 강팀끼리 붙으면 확률이 1을 넘고
    약팀끼리 붙으면 1에 못 미친다. 즉 잔여 일정이 강팀에 몰린 팀은
    승수가 체계적으로 과대 추정된다. 일정 난이도가 다른 팀을 비교하는
    것이 이 프로그램의 목적이므로, 이 결함은 결과를 무의미하게 만든다.
""")

print("=" * 64)
print("  결함 2 — 동률 처리 누락 (검증 불변식 위반)")
print("=" * 64)
print("""
  초기 구현: 5번째로 높은 승률을 구하고, 그 이상인 팀을 모두 진출 처리.
  동률이 발생하면 6팀 이상이 선택되지만 예외 없이 실행된다.
""")

rng = np.random.default_rng(7)
N = 200_000
# 최종 승패를 직접 표집해 동률 상황을 재현
w = rng.binomial(144, np.clip(talent, 0, 1), size=(N, len(teams)))
pct = w / 144

cutoff = np.sort(pct, axis=1)[:, -PLAYOFF_SPOTS]
made_buggy = pct >= cutoff[:, None]              # 동률 시 6팀 이상 선택
prob_buggy = made_buggy.mean(axis=0)

jitter = rng.random((N, len(teams))) * 1e-9
order = np.argsort(-(pct + jitter), axis=1, kind="stable")
made_fixed = np.zeros((N, len(teams)), dtype=bool)
np.put_along_axis(made_fixed, order[:, :PLAYOFF_SPOTS], True, axis=1)
prob_fixed = made_fixed.mean(axis=0)

n_over = (made_buggy.sum(axis=1) > PLAYOFF_SPOTS).mean()

print(f"  진출 팀이 5팀을 초과한 시뮬레이션 비율 : {n_over*100:.1f}%")
print(f"  진출확률 총합 (수정 전) : {prob_buggy.sum():.4f}   ← 5.0이어야 함")
print(f"  진출확률 총합 (수정 후) : {prob_fixed.sum():.4f}   ← PASS")

diff = pd.DataFrame({
    "team": teams,
    "수정전(%)": prob_buggy * 100,
    "수정후(%)": prob_fixed * 100,
})
diff["차이(%p)"] = diff["수정전(%)"] - diff["수정후(%)"]
print("\n" + diff.sort_values("수정후(%)", ascending=False)
      .to_string(index=False, float_format=lambda x: f"{x:6.2f}"))

print("""
  → 매 시뮬레이션마다 정확히 5팀이 진출하므로, 진출 확률의 총합은
    반드시 5.0이 된다. 이 불변식은 정답을 몰라도 계산 가능하기 때문에
    결과를 자체 검증할 수 있는 기준이 된다.
    총합이 5를 넘었다는 것은 순위 결정 로직에 오류가 있다는 신호였고,
    추적한 결과 동률 처리 누락이 원인이었다.
""")
