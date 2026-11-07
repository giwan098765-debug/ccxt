"""재무제표 평가 — 밸류에이션·수익성·성장성·안정성 점수(0~100)와 근거."""
from __future__ import annotations

import math


def _ok(x) -> bool:
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def score_fundamentals(f: dict) -> dict:
    if not f:
        return {"score": None, "points": [], "warnings": ["재무 데이터 없음"]}
    pts, warns, parts = [], [], []
    per, pbr, roe = f.get("per"), f.get("pbr"), f.get("roe")
    if _ok(per):
        if 0 < per < 10:
            parts.append(90); pts.append(f"PER {per:.1f}배 — 저평가")
        elif per < 20:
            parts.append(65)
        elif per < 40:
            parts.append(40)
        else:
            parts.append(20); warns.append(f"PER {per:.0f}배 — 고평가 부담")
    if _ok(pbr):
        if 0 < pbr < 1:
            parts.append(85); pts.append(f"PBR {pbr:.2f}배 — 청산가치 이하")
        elif pbr < 2.5:
            parts.append(60)
        else:
            parts.append(35)
    if _ok(roe):
        if roe > 0.15:
            parts.append(90); pts.append(f"ROE {roe*100:.1f}% — 고수익성")
        elif roe > 0.08:
            parts.append(65)
        elif roe > 0:
            parts.append(45)
        else:
            parts.append(15); warns.append(f"ROE {roe*100:.1f}% — 적자")
    for key, label in (("revenue_growth", "매출 성장률"), ("earnings_growth", "이익 성장률")):
        g = f.get(key)
        if _ok(g):
            if g > 0.3:
                parts.append(95); pts.append(f"{label} {g*100:+.0f}% — 고성장")
            elif g > 0.1:
                parts.append(70); pts.append(f"{label} {g*100:+.0f}%")
            elif g > 0:
                parts.append(50)
            else:
                parts.append(25); warns.append(f"{label} {g*100:+.0f}% — 역성장")
    om = f.get("op_margin")
    if _ok(om):
        parts.append(85 if om > 0.15 else 60 if om > 0.05 else 30 if om > 0 else 10)
        if om > 0.15:
            pts.append(f"영업이익률 {om*100:.1f}%")
        elif om <= 0:
            warns.append("영업적자")
    de = f.get("debt_to_equity")
    if _ok(de):
        parts.append(80 if de < 50 else 60 if de < 100 else 35 if de < 200 else 15)
        if de >= 200:
            warns.append(f"부채비율 {de:.0f}% — 재무위험")
    q = f.get("op_income_q")
    if q and len(q) >= 2 and all(_ok(v) for v in q[-2:]):
        if q[-2] <= 0 < q[-1]:
            parts.append(90); pts.append("최근 분기 영업이익 흑자전환")
        elif q[-2] > 0 and q[-1] > q[-2] * 1.3:
            parts.append(85); pts.append(f"최근 분기 영업이익 전분기 대비 {q[-1]/q[-2]*100-100:+.0f}%")
    score = round(sum(parts) / len(parts)) if parts else None
    return {"score": score, "points": pts, "warnings": warns}
