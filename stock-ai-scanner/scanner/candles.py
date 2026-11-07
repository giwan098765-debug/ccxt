"""캔들 패턴 인식 (벡터화). 각 패턴은 bool Series 로 반환되어 과거 통계 산출에도 사용된다."""
from __future__ import annotations

import pandas as pd

# 이름, 방향(+1 상승 / -1 하락), 설명
PATTERN_INFO = {
    "hammer": ("망치형", 1, "하락 후 긴 아래꼬리 — 저가 매수세 유입, 반등 신호"),
    "inv_hammer": ("역망치형", 1, "하락 후 긴 위꼬리 — 매수 시도 시작"),
    "bull_engulf": ("상승 장악형", 1, "전일 음봉을 감싸는 양봉 — 강한 추세 전환 신호"),
    "piercing": ("관통형", 1, "전일 음봉 몸통 절반 이상 회복"),
    "morning_star": ("샛별형", 1, "하락-도지-양봉 3봉 반전 — 바닥 신호"),
    "three_soldiers": ("적삼병", 1, "3일 연속 양봉 상승 — 추세 상승 초입"),
    "long_bull": ("장대양봉", 1, "몸통이 평균 2배 이상인 양봉 — 세력 매수 흔적"),
    "bull_harami": ("상승 잉태형", 1, "큰 음봉 안의 작은 양봉 — 하락 힘 약화"),
    "doji": ("도지", 0, "매수·매도 균형 — 방향 전환 가능 구간"),
    "shooting_star": ("유성형", -1, "상승 후 긴 위꼬리 — 고점 매도 압력"),
    "bear_engulf": ("하락 장악형", -1, "전일 양봉을 감싸는 음봉 — 하락 전환 경고"),
    "dark_cloud": ("흑운형", -1, "전일 양봉 절반 이상 반납"),
    "evening_star": ("석별형", -1, "상승-도지-음봉 3봉 반전 — 천장 신호"),
    "long_bear": ("장대음봉", -1, "몸통이 평균 2배 이상인 음봉 — 대량 매도"),
}


def detect_candles(df: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    body = (c - o).abs()
    rng = (h - l).replace(0, 1e-9)
    upper = h - pd.concat([o, c], axis=1).max(axis=1)
    lower = pd.concat([o, c], axis=1).min(axis=1) - l
    avg_body = body.rolling(20, min_periods=5).mean()
    bull, bear = c > o, c < o
    downtrend = c.shift(1) < c.rolling(10).mean().shift(1)
    uptrend = c.shift(1) > c.rolling(10).mean().shift(1)
    po, pc = o.shift(1), c.shift(1)
    pbody = body.shift(1)
    out = pd.DataFrame(index=df.index)
    out["doji"] = body <= rng * 0.1
    out["hammer"] = downtrend & (lower >= 2 * body) & (upper <= body * 0.6 + rng * 0.05) & (body > 0)
    out["inv_hammer"] = downtrend & (upper >= 2 * body) & (lower <= body * 0.6 + rng * 0.05) & (body > 0)
    out["shooting_star"] = uptrend & (upper >= 2 * body) & (lower <= body * 0.6 + rng * 0.05) & (body > 0)
    out["bull_engulf"] = bull & (pc < po) & (c >= po) & (o <= pc) & (body > pbody)
    out["bear_engulf"] = bear & (pc > po) & (o >= pc) & (c <= po) & (body > pbody)
    out["piercing"] = bull & (pc < po) & (o < pc) & (c > (po + pc) / 2) & (c < po)
    out["dark_cloud"] = bear & (pc > po) & (o > pc) & (c < (po + pc) / 2) & (c > po)
    small_mid = body.shift(1) <= avg_body.shift(1) * 0.4
    out["morning_star"] = (c.shift(2) < o.shift(2)) & (body.shift(2) > avg_body.shift(2)) & small_mid & bull & (c > (o.shift(2) + c.shift(2)) / 2)
    out["evening_star"] = (c.shift(2) > o.shift(2)) & (body.shift(2) > avg_body.shift(2)) & small_mid & bear & (c < (o.shift(2) + c.shift(2)) / 2)
    out["three_soldiers"] = bull & bull.shift(1, fill_value=False) & bull.shift(2, fill_value=False) & (c > c.shift(1)) & (c.shift(1) > c.shift(2)) & (body > avg_body * 0.6)
    out["long_bull"] = bull & (body >= 2 * avg_body)
    out["long_bear"] = bear & (body >= 2 * avg_body)
    out["bull_harami"] = bull & (pc < po) & (pbody > avg_body) & (o > pc) & (c < po)
    return out.fillna(False).astype(bool)


def recent_patterns(candles: pd.DataFrame, lookback: int = 3) -> list[dict]:
    """최근 lookback 봉 내 발생한 패턴 목록 (최신순)."""
    res = []
    tail = candles.iloc[-lookback:]
    for i in range(len(tail) - 1, -1, -1):
        row = tail.iloc[i]
        ago = len(tail) - 1 - i
        for key, hit in row.items():
            if hit:
                name, bias, desc = PATTERN_INFO[key]
                res.append({"key": key, "name": name, "bias": bias, "desc": desc, "days_ago": ago, "date": tail.index[i]})
    return res
