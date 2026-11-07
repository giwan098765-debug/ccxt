"""상단(오버레이) 지표와 보조지표 계산."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100.0).where(dn.notna())


def _mean_abs_dev(s: pd.Series, n: int) -> pd.Series:
    a = s.to_numpy(dtype=float)
    res = np.full(len(a), np.nan)
    if len(a) >= n:
        w = np.lib.stride_tricks.sliding_window_view(a, n)
        res[n - 1:] = np.abs(w - w.mean(axis=1, keepdims=True)).mean(axis=1)
    return pd.Series(res, index=s.index).replace(0, np.nan)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV 데이터프레임에 모든 지표 컬럼을 추가해 새 데이터프레임을 반환."""
    out = df.copy()
    c, h, l, v = out["Close"], out["High"], out["Low"], out["Volume"]

    # ── 상단 지표 (가격 차트 위에 그리는 지표) ──
    for n in (5, 20, 60, 120, 240):
        out[f"ma{n}"] = c.rolling(n, min_periods=n).mean()
    out["ema12"] = c.ewm(span=12, adjust=False).mean()
    out["ema26"] = c.ewm(span=26, adjust=False).mean()
    out["bb_mid"] = c.rolling(20).mean()
    std = c.rolling(20).std(ddof=0)
    out["bb_up"] = out["bb_mid"] + 2 * std
    out["bb_low"] = out["bb_mid"] - 2 * std
    out["bb_width"] = (out["bb_up"] - out["bb_low"]) / out["bb_mid"]
    out["bb_pos"] = (c - out["bb_low"]) / (out["bb_up"] - out["bb_low"]).replace(0, np.nan)
    # 일목균형표
    out["ichi_conv"] = (h.rolling(9).max() + l.rolling(9).min()) / 2
    out["ichi_base"] = (h.rolling(26).max() + l.rolling(26).min()) / 2
    out["ichi_span_a"] = ((out["ichi_conv"] + out["ichi_base"]) / 2).shift(26)
    out["ichi_span_b"] = ((h.rolling(52).max() + l.rolling(52).min()) / 2).shift(26)
    # 엔벨로프 대신 20일 최고/최저 (박스권)
    out["hh20"] = h.rolling(20).max()
    out["ll20"] = l.rolling(20).min()
    out["hh60"] = h.rolling(60).max()
    out["hh250"] = h.rolling(250, min_periods=120).max()

    # ── 보조지표 ──
    out["rsi"] = _rsi(c, 14)
    out["macd"] = out["ema12"] - out["ema26"]
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]
    ll14, hh14 = l.rolling(14).min(), h.rolling(14).max()
    out["stoch_k"] = 100 * (c - ll14) / (hh14 - ll14).replace(0, np.nan)
    out["stoch_d"] = out["stoch_k"].rolling(3).mean()
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    out["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    out["atr_pct"] = out["atr"] / c
    up_move, dn_move = h.diff(), -l.diff()
    plus_dm = np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0)
    atr_s = tr.ewm(alpha=1 / 14, adjust=False).mean().replace(0, np.nan)
    out["plus_di"] = 100 * pd.Series(plus_dm, index=out.index).ewm(alpha=1 / 14, adjust=False).mean() / atr_s
    out["minus_di"] = 100 * pd.Series(minus_dm, index=out.index).ewm(alpha=1 / 14, adjust=False).mean() / atr_s
    dx = 100 * (out["plus_di"] - out["minus_di"]).abs() / (out["plus_di"] + out["minus_di"]).replace(0, np.nan)
    out["adx"] = dx.ewm(alpha=1 / 14, adjust=False).mean()
    out["obv"] = (np.sign(c.diff()).fillna(0) * v).cumsum()
    out["obv_slope20"] = (out["obv"] - out["obv"].shift(20)) / (v.rolling(20).mean() * 20).replace(0, np.nan)
    tp = (h + l + c) / 3
    mf = tp * v
    pos = mf.where(tp > tp.shift(1), 0.0).rolling(14).sum()
    neg = mf.where(tp < tp.shift(1), 0.0).rolling(14).sum()
    out["mfi"] = 100 - 100 / (1 + pos / neg.replace(0, np.nan))
    out["cci"] = (tp - tp.rolling(20).mean()) / (0.015 * _mean_abs_dev(tp, 20))
    out["vol_ma20"] = v.rolling(20).mean()
    out["vol_ratio"] = v / v.shift(1).rolling(20).mean().replace(0, np.nan)
    out["value"] = c * v  # 거래대금
    out["ret1"] = c.pct_change()
    out["ret5"] = c.pct_change(5)
    out["ret20"] = c.pct_change(20)
    out["bb_width_pct"] = out["bb_width"].rolling(120, min_periods=60).rank(pct=True)
    return out
