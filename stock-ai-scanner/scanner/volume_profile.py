"""매물대(Volume Profile) 분석 — 가격대별 누적 거래량으로 지지/저항을 찾는다."""
from __future__ import annotations

import numpy as np
import pandas as pd


def volume_profile(df: pd.DataFrame, lookback: int = 120, bins: int = 30) -> dict:
    d = df.iloc[-lookback:]
    lo, hi = float(d["Low"].min()), float(d["High"].max())
    if not np.isfinite(lo) or hi <= lo:
        return {}
    edges = np.linspace(lo, hi, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    vol = np.zeros(bins)
    for bl, bh, bv in zip(d["Low"].values, d["High"].values, d["Volume"].values):
        i0 = int(np.clip(np.searchsorted(edges, bl, side="right") - 1, 0, bins - 1))
        i1 = int(np.clip(np.searchsorted(edges, bh, side="right") - 1, 0, bins - 1))
        vol[i0:i1 + 1] += bv / (i1 - i0 + 1)
    price = float(d["Close"].iloc[-1])
    total = vol.sum() or 1.0
    poc_i = int(vol.argmax())
    # 가치영역(Value Area 70%)
    order = np.argsort(vol)[::-1]
    acc, va = 0.0, []
    for i in order:
        va.append(i)
        acc += vol[i]
        if acc >= 0.7 * total:
            break
    hvn_thr = np.percentile(vol, 70)
    cur_i = int(np.clip(np.searchsorted(edges, price, side="right") - 1, 0, bins - 1))
    above = [i for i in range(cur_i + 1, bins) if vol[i] >= hvn_thr]
    below = [i for i in range(cur_i - 1, -1, -1) if vol[i] >= hvn_thr]
    overhead = float(vol[cur_i + 1:].sum() / total)
    support_ratio = float(vol[:cur_i].sum() / total)
    return {
        "centers": centers, "volumes": vol, "edges": edges,
        "poc": float(centers[poc_i]),
        "va_low": float(edges[min(va)]), "va_high": float(edges[max(va) + 1]),
        "resistance": float(centers[above[0]]) if above else None,
        "support": float(centers[below[0]]) if below else None,
        "overhead_supply": overhead,      # 현재가 위에 쌓인 매물 비율 (낮을수록 상승 저항 적음)
        "support_volume": support_ratio,  # 현재가 아래 매물 비율 (높을수록 지지 탄탄)
        "price": price,
        "above_poc": price > centers[poc_i],
    }
