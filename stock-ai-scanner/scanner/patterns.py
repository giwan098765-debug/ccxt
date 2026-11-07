"""차트 모양/셋업(매매 자리) 인식.

각 셋업은 과거 전 구간에서 bool Series 로 계산되므로, '이 자리가 과거에 몇 번 나왔고
그때 몇 % 올랐는지'를 그대로 백테스트할 수 있다 (probability.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# key: (한글명, 설명)
SETUPS = {
    "breakout": ("돌파형", "60일 신고가를 거래량 동반(평균 1.8배↑)으로 돌파"),
    "rebound": ("반등형", "RSI 과매도(32↓) 후 상승 전환 + 양봉 — 기술적 반등 자리"),
    "trend": ("추세추종형", "이평 정배열(5>20>60>120) + ADX 22↑ 강한 상승 추세"),
    "pullback": ("눌림목형", "상승 추세 중 20일선 근처 조정 + 거래량 감소 — 재상승 대기 자리"),
    "squeeze": ("수렴(스퀴즈)형", "볼린저밴드 폭이 120일 중 하위 12% — 큰 시세 분출 임박"),
    "volume": ("거래량 급증형", "평균 대비 3배↑ 거래량 + 양봉 — 세력/수급 유입"),
    "golden_cross": ("골든크로스", "5일선이 20일선을 상향 돌파"),
    "macd_turn": ("MACD 전환", "0선 아래에서 MACD 가 시그널선을 상향 돌파"),
}


def setup_signals(d: pd.DataFrame) -> pd.DataFrame:
    """지표가 붙은 데이터프레임(add_indicators 결과)에서 셋업 bool 시계열 생성."""
    c, o = d["Close"], d["Open"]
    s = pd.DataFrame(index=d.index)
    s["breakout"] = (c > d["hh60"].shift(1)) & (d["vol_ratio"] >= 1.8) & (d["ret1"] > 0)
    s["rebound"] = (d["rsi"].rolling(5).min() < 32) & (d["rsi"] > d["rsi"].shift(1)) & (c > o) & (c > c.shift(1))
    s["trend"] = (d["ma5"] > d["ma20"]) & (d["ma20"] > d["ma60"]) & (d["ma60"] > d["ma120"]) & (d["adx"] > 22) & (c > d["ma20"])
    s["pullback"] = ((d["ma20"] > d["ma60"]) & (d["ma60"] > d["ma60"].shift(5)) & ((c / d["ma20"] - 1).abs() < 0.03)
                     & (d["vol_ratio"] < 0.9) & (d["ret20"] > 0))
    s["squeeze"] = (d["bb_width_pct"] < 0.12) & (c > d["ma60"])
    s["volume"] = (d["vol_ratio"] >= 3) & (c > o)
    s["golden_cross"] = (d["ma5"] > d["ma20"]) & (d["ma5"].shift(1) <= d["ma20"].shift(1))
    s["macd_turn"] = (d["macd"] > d["macd_signal"]) & (d["macd"].shift(1) <= d["macd_signal"].shift(1)) & (d["macd"] < 0)
    return s.fillna(False).astype(bool)


def detect_double_bottom(d: pd.DataFrame, lookback: int = 80, tol: float = 0.04) -> dict | None:
    """최근 lookback 봉 안의 쌍바닥(W) 탐지 — 현재 시점 전용."""
    x = d.iloc[-lookback:]
    lows = x["Low"].values
    if len(lows) < 30:
        return None
    # 지역 저점(좌우 5봉 최저)
    piv = [i for i in range(5, len(lows) - 3) if lows[i] == lows[max(0, i - 5):i + 6].min()]
    for a in range(len(piv) - 1, 0, -1):
        for b in range(a - 1, -1, -1):
            i1, i2 = piv[b], piv[a]
            if i2 - i1 < 10:
                continue
            l1, l2 = lows[i1], lows[i2]
            if abs(l2 / l1 - 1) > tol:
                continue
            neck = x["High"].values[i1:i2].max()
            if neck / max(l1, l2) - 1 < 0.06:
                continue
            price = x["Close"].values[-1]
            return {
                "first": x.index[i1], "second": x.index[i2], "low": float(min(l1, l2)), "neckline": float(neck),
                "breakout": bool(price > neck), "target": float(neck + (neck - min(l1, l2))),
            }
    return None


def chart_shape(d: pd.DataFrame) -> dict:
    """현재 차트 모양 요약."""
    last = d.iloc[-1]
    c = float(last["Close"])
    ma = {n: last.get(f"ma{n}") for n in (5, 20, 60, 120)}
    shape: dict = {}
    if all(pd.notna(v) for v in ma.values()):
        if ma[5] > ma[20] > ma[60] > ma[120]:
            shape["alignment"] = "정배열 (강세)"
        elif ma[5] < ma[20] < ma[60] < ma[120]:
            shape["alignment"] = "역배열 (약세)"
        else:
            shape["alignment"] = "혼조 (전환 구간)"
    hh250 = last.get("hh250")
    if pd.notna(hh250):
        shape["from_52w_high"] = c / hh250 - 1
    rng20 = (last["hh20"] - last["ll20"]) / c if pd.notna(last["hh20"]) else np.nan
    shape["box20_range"] = rng20
    if pd.notna(rng20) and rng20 < 0.12:
        shape["box"] = "좁은 박스권 횡보 (에너지 응축)"
    slope = (d["ma20"].iloc[-1] / d["ma20"].iloc[-6] - 1) if len(d) > 25 else np.nan
    shape["ma20_slope5"] = slope
    if pd.notna(last.get("ichi_span_a")) and pd.notna(last.get("ichi_span_b")):
        top = max(last["ichi_span_a"], last["ichi_span_b"])
        bot = min(last["ichi_span_a"], last["ichi_span_b"])
        shape["ichimoku"] = "구름대 위 (강세)" if c > top else ("구름대 아래 (약세)" if c < bot else "구름대 안 (중립)")
    db = detect_double_bottom(d)
    if db:
        shape["double_bottom"] = db
    return shape
