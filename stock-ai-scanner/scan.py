"""터미널 스캔.  예) python scan.py --market KOSDAQ --limit 500"""
from __future__ import annotations

import argparse

import pandas as pd

from scanner.engine import CATEGORIES, ScanConfig, Scanner


def main():
    ap = argparse.ArgumentParser(description="AI 주식 스캐너")
    ap.add_argument("--market", default="KRX", choices=["KRX", "KOSPI", "KOSDAQ", "US"])
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--surge", type=float, default=15, help="이번주 급등주 기준 %%")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--demo", action="store_true", help="합성 데이터(오프라인)")
    ap.add_argument("--min-value", type=float, default=None, help="최소 20일 평균 거래대금")
    a = ap.parse_args()
    cfg = ScanConfig(market=a.market, limit=a.limit, surge_min=a.surge / 100, demo=a.demo)
    if a.min_value is not None:
        cfg.min_value = a.min_value
    elif a.market == "US":
        cfg.min_value = 5e6
    res = Scanner(cfg).run(lambda p, m: print(f"\r[{p*100:5.1f}%] {m:<40}", end="", flush=True))
    print()
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    mm = res.model_metrics
    print(f"데이터: {res.provider_name} | 분석 {len(res.summary)}종목")
    if mm.get("trained"):
        print(f"AI 모델 검증 AUC {mm['auc_test']:.3f} · 상위10% 적중률 {mm['top10_hit_rate_test']*100:.1f}% (평균 {mm['base_rate']*100:.1f}%)")
    cols = ["name", "close", "week_ret", "ai_prob", "success_prob", "best_setup", "target_pct", "stop_pct", "ev", "composite"]
    for key, title in CATEGORIES.items():
        df = res.categories[key]
        print(f"\n{title} ({len(df)})")
        if df.empty:
            continue
        t = df[cols].head(a.top).copy()
        for c in ("week_ret", "ai_prob", "success_prob", "target_pct", "stop_pct", "ev"):
            t[c] = (t[c] * 100).round(1)
        print(t.to_string())
        if key == "surge_week":
            for code in df.index[:3]:
                print(f"  ▶ {res.details[code]['row']['name']}")
                for line in res.details[code]["surge_reasons"]:
                    print(f"     {line}")


if __name__ == "__main__":
    main()
