"""
KBO 가을야구 진출 확률 시뮬레이터
==================================

잔여 경기를 몬테카를로로 반복 시뮬레이션해 상위 5위 진입 확률을 추정한다.

설계 원칙
---------
1. 승률 추정은 단순 시즌 승률이 아니라 log5(Bradley-Terry) 모델을 사용해
   '상대 팀 전력'을 반영한다. 단순 승률 모델은 잔여 일정 난이도를 무시하므로
   약팀과 많이 붙는 팀의 확률을 체계적으로 과소평가한다.
2. 모든 추정치는 검증 불변식(invariant)과 표준오차를 함께 출력한다.
   검증 없는 시뮬레이션 결과는 신뢰하지 않는다.
"""

import numpy as np
import pandas as pd

# --- 리그 상수 (KBO 2026) ---------------------------------------------------
TOTAL_GAMES = 144       # 팀당 정규시즌 경기 수
PLAYOFF_SPOTS = 5       # 가을야구 진출 팀 수 (와일드카드 포함 상위 5팀)
P_TIE = 0.022           # 무승부 발생률 (2026시즌 실측: 12무 / 543경기)
HFA_ODDS = 1.15         # 홈 어드밴티지 오즈비 (홈 승률 ≈ 53.5% 기준)


# --- 1. 승률 모델 -----------------------------------------------------------
def log5(p_a: np.ndarray, p_b: np.ndarray) -> np.ndarray:
    """
    log5 (Bill James) — 승률 p_a 팀이 승률 p_b 팀을 이길 확률.

        P = (p_a - p_a*p_b) / (p_a + p_b - 2*p_a*p_b)

    Bradley-Terry 모델과 수학적으로 동치이며, 두 팀이 모두 리그 평균(0.5)일 때
    0.5를 반환하고 상대가 강할수록 승률이 낮아진다.
    """
    num = p_a - p_a * p_b
    den = p_a + p_b - 2 * p_a * p_b
    return np.divide(num, den, out=np.full_like(num, 0.5), where=den != 0)


def game_probabilities(p_home: np.ndarray, p_away: np.ndarray):
    """
    경기별 (홈승, 무, 원정승) 확률을 반환한다.
    홈 어드밴티지는 log5 결과를 오즈로 변환해 HFA_ODDS를 곱하는 방식으로 반영.
    """
    base = log5(p_home, p_away)                       # 중립 구장 기준 홈팀 승률
    odds = base / (1 - base) * HFA_ODDS               # 홈 어드밴티지 적용
    p_h = odds / (1 + odds)

    # 무승부 확률만큼 승/패 확률을 비례 축소
    p_h *= (1 - P_TIE)
    p_a = (1 - P_TIE) - p_h
    return p_h, p_a


# --- 2. 몬테카를로 시뮬레이션 -----------------------------------------------
def simulate(standings: pd.DataFrame,
             schedule: pd.DataFrame,
             n_sims: int = 100_000,
             seed: int = 42):
    """
    잔여 경기를 n_sims회 시뮬레이션한다.

    Returns
    -------
    prob : (T,) 각 팀의 상위 5위 진입 확률
    made : (n_sims, T) bool. 시뮬레이션별 진출 여부
    outcome : (n_sims, G) int8. 0=홈승, 1=원정승, 2=무 (레버리지 분석용)
    teams : 팀 이름 리스트 (인덱스 순서)
    """
    teams = standings["team"].tolist()
    idx = {t: i for i, t in enumerate(teams)}
    T = len(teams)

    base_w = standings["w"].to_numpy(np.int32)
    base_l = standings["l"].to_numpy(np.int32)

    # 현재 시즌 승률을 팀 실력(talent) 추정치로 사용 (무승부 제외)
    talent = base_w / (base_w + base_l)

    home = schedule["home"].map(idx).to_numpy()
    away = schedule["away"].map(idx).to_numpy()
    G = len(schedule)

    p_h, p_a = game_probabilities(talent[home], talent[away])

    rng = np.random.default_rng(seed)
    u = rng.random((n_sims, G), dtype=np.float32)

    # 0=홈승, 1=원정승, 2=무
    outcome = np.where(u < p_h, 0, np.where(u < p_h + p_a, 1, 2)).astype(np.int8)
    del u

    wins = np.tile(base_w, (n_sims, 1))
    losses = np.tile(base_l, (n_sims, 1))

    for g in range(G):
        h, a = home[g], away[g]
        hw = outcome[:, g] == 0
        aw = outcome[:, g] == 1
        wins[:, h] += hw
        losses[:, a] += hw
        wins[:, a] += aw
        losses[:, h] += aw

    pct = wins / (wins + losses)

    # 동률 처리: 미세 난수를 더해 순위를 강제로 확정한다.
    # jitter(1e-9)는 서로 다른 승률의 최소 간격(≈1/144² ≈ 4.8e-5)보다 훨씬 작으므로
    # 실제 승률 차이는 절대 뒤집지 않고 '완전 동률'만 무작위로 가른다.
    jitter = rng.random((n_sims, T)) * 1e-9
    order = np.argsort(-(pct + jitter), axis=1, kind="stable")

    made = np.zeros((n_sims, T), dtype=bool)
    np.put_along_axis(made, order[:, :PLAYOFF_SPOTS], True, axis=1)

    return made.mean(axis=0), made, outcome, teams


# --- 3. 검증 ----------------------------------------------------------------
def validate(prob: np.ndarray, n_sims: int, schedule: pd.DataFrame,
             standings: pd.DataFrame) -> list[str]:
    """
    시뮬레이션 결과의 논리적 타당성을 검사한다.
    하나라도 실패하면 결과를 신뢰하지 않는다.
    """
    msgs = []

    # [불변식 1] 모든 팀의 진출 확률 합 = 정확히 5.0
    # 매 시뮬레이션마다 정확히 5팀이 진출하므로 기댓값의 합은 반드시 5가 된다.
    # 이 값이 5에서 벗어나면 순위 결정 로직(특히 동률 처리)에 버그가 있다는 뜻.
    total = prob.sum()
    ok = abs(total - PLAYOFF_SPOTS) < 1e-9
    msgs.append(f"[{'PASS' if ok else 'FAIL'}] 진출확률 총합 = {total:.9f} (기댓값 5.0)")

    # [불변식 2] 잔여 경기 수 정합성: 팀별 잔여 경기 합 = 총 잔여 경기 × 2
    played = standings["w"] + standings["d"] + standings["l"]
    remain_expected = (TOTAL_GAMES - played).sum()
    remain_actual = len(schedule) * 2
    ok = remain_expected == remain_actual
    msgs.append(f"[{'PASS' if ok else 'FAIL'}] 잔여 경기 정합성 = "
                f"{remain_actual} (순위표 기준 {remain_expected})")

    # [불변식 3] 확률 범위
    ok = bool(((prob >= 0) & (prob <= 1)).all())
    msgs.append(f"[{'PASS' if ok else 'FAIL'}] 모든 확률이 [0,1] 범위 내")

    # [정밀도] 저확률 추정의 상대표준오차 — 낮은 확률일수록 더 많은 시행이 필요
    p = prob[prob > 0].min() if (prob > 0).any() else 0
    if p > 0:
        rse = np.sqrt((1 - p) / (p * n_sims)) * 100
        flag = "WARN" if rse > 10 else "PASS"
        msgs.append(f"[{flag}] 최저 확률({p*100:.2f}%)의 상대표준오차 = {rse:.1f}% "
                    f"(n={n_sims:,})")
    return msgs


# --- 4. 레버리지(민감도) 분석 -----------------------------------------------
def leverage(made, outcome, schedule, teams, target="롯데"):
    """
    잔여 경기별 '영향도'를 계산한다.

        leverage = P(진출 | 해당 경기 승) - P(진출 | 해당 경기 패)

    같은 시뮬레이션 집합을 경기 결과로 조건부 분할해 추정하므로 추가 연산이 없다.
    DOE의 주효과(main effect) 추정과 구조적으로 동일하다.
    """
    t = teams.index(target)
    rows = []

    mask = (schedule["home"] == target) | (schedule["away"] == target)
    for g in np.where(mask.to_numpy())[0]:
        is_home = schedule["home"].iloc[g] == target
        won = outcome[:, g] == (0 if is_home else 1)
        lost = outcome[:, g] == (1 if is_home else 0)

        if won.sum() < 100 or lost.sum() < 100:
            continue

        p_w = made[won, t].mean()
        p_l = made[lost, t].mean()
        opp = schedule["away"].iloc[g] if is_home else schedule["home"].iloc[g]

        rows.append({
            "date": schedule["date"].iloc[g],
            "opponent": opp,
            "venue": "홈" if is_home else "원정",
            "p_if_win": p_w,
            "p_if_lose": p_l,
            "leverage": p_w - p_l,
        })

    return pd.DataFrame(rows).sort_values("leverage", ascending=False)
