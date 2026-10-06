"""Fixed 20% drawdown/covariance/cushion experiment; offline spot replay only."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.research_execution_portfolio import load_daily, VALIDATION, TRANSFER, END
from spot_bot.backtest.portfolio import run_portfolio_backtest
from spot_bot.portfolio.risk import PortfolioRisk
from spot_bot.portfolio.trend import TrendPortfolio


LIMIT = .20
BEGIN = pd.Timestamp("2022-01-01", tz="UTC")
GAP_DATE = pd.Timestamp("2024-03-11", tz="UTC")
VARIANTS = {
    "static_50": (.5, None), "static_75": (.75, None),
    "vol_15": (1., PortfolioRisk(volatility_target=.15)),
    "vol_25": (1., PortfolioRisk(volatility_target=.25)),
    "cushion_3": (1., PortfolioRisk(drawdown_limit=LIMIT, cushion_multiplier=3)),
    "cushion_5": (1., PortfolioRisk(drawdown_limit=LIMIT, cushion_multiplier=5)),
    "vol_25_cushion_5": (1., PortfolioRisk(volatility_target=.25, drawdown_limit=LIMIT,
                                          cushion_multiplier=5)),
}
COSTS = dict(fee_rate=.001, slippage_bps=5., spread_bps=2.)
STRESS_COSTS = dict(fee_rate=.002, slippage_bps=10., spread_bps=2.)


def _replay(markets, name, start, end, out, label, costs=COSTS):
    cap, risk = VARIANTS[name]
    policy = TrendPortfolio("momentum_vol", max_exposure=cap, asset_cap=cap / 2)
    equity, trades, summary = run_portfolio_backtest(markets, policy, risk=risk,
                                                    start=start, end=end, **costs)
    # Check actual paths, not just aggregate output metrics.
    if (equity.usdt.min() < -1e-8 or equity.filter(like="base_").min().min() < -1e-8):
        raise AssertionError("Negative spot cash or inventory")
    dates = pd.DatetimeIndex(equity.timestamp)
    nav = equity.usdt.to_numpy().copy()
    for symbol, frame in markets.items():
        nav += equity[f"base_{symbol}"].to_numpy() * frame.loc[dates, "close"].to_numpy()
    np.testing.assert_allclose(nav, equity.equity.to_numpy(), rtol=1e-12, atol=1e-8)
    np.testing.assert_allclose([summary["fees_paid_total"], summary["slippage_paid_total"]],
                               [trades.fee.sum(), trades.slippage.sum()], rtol=1e-12, atol=1e-8)
    summary["cagr"] = (summary["final_equity"] / 1000) ** (365 / len(equity)) - 1
    summary["drawdown_within_20pct_bound"] = summary["max_intraday_drawdown_bound"] >= -LIMIT
    equity.to_csv(out / f"{name}_{label}_equity.csv", index=False)
    trades.to_csv(out / f"{name}_{label}_trades.csv", index=False)
    print(f"{name} {label}: net={summary['total_return']:+.3%}, "
          f"closeDD={summary['maxDD']:.3%}, boundDD={summary['max_intraday_drawdown_bound']:.3%}", flush=True)
    return summary


def research(markets, out):
    out.mkdir(parents=True, exist_ok=True)
    development = {name: _replay(markets, name, BEGIN, VALIDATION, out, "development")
                   for name in VARIANTS}
    eligible = [name for name, s in development.items()
                if s["net_pnl"] > 0 and s["drawdown_within_20pct_bound"]]
    frozen = max(eligible, key=lambda name: development[name]["net_pnl"]) if eligible else "cash"
    selection = {"frozen_candidate": frozen, "development": development,
                 "drawdown_limit": LIMIT,
                 "criterion": "maximum positive 2022-2024 continuous net P&L with conservative drawdown <=20%"}
    (out / "frozen_risk_selection.json").write_text(json.dumps(selection, indent=2, allow_nan=False) + "\n")
    print(f"Frozen before subsequent-period runs: {frozen}", flush=True)
    gap_markets = {s: f.copy() for s, f in markets.items()}
    for frame in gap_markets.values():
        frame.loc[frame.index >= GAP_DATE, ["open", "high", "low", "close"]] *= .6
    periods = {"2025": (VALIDATION, TRANSFER), "2026_01_09": (TRANSFER, END),
               "continuous": (BEGIN, END)}
    scenarios = {}
    for name, (cap, risk) in VARIANTS.items():
        scenario = {"policy": asdict(TrendPortfolio("momentum_vol", max_exposure=cap, asset_cap=cap / 2)),
                    "risk": asdict(risk) if risk else None, "development": development[name],
                    "periods": {}, "doubled_costs": {}}
        for label, (start, end) in periods.items():
            scenario["periods"][label] = _replay(markets, name, start, end, out, label)
            scenario["doubled_costs"][label] = _replay(markets, name, start, end, out,
                                                      f"{label}_double_cost", STRESS_COSTS)
        scenario["correlated_40pct_gap"] = _replay(gap_markets, name, BEGIN, END, out, "gap")
        summaries = [scenario["development"], *scenario["periods"].values(),
                     *scenario["doubled_costs"].values()]
        scenario["all_historical_drawdowns_within_20pct_bound"] = all(
            s["drawdown_within_20pct_bound"] for s in summaries)
        scenarios[name] = scenario
    gates = {"development_eligible": frozen != "cash"}
    if frozen != "cash":
        chosen = scenarios[frozen]
        gates.update(historical_drawdown_limit=chosen["all_historical_drawdowns_within_20pct_bound"],
                     later_nominal_positive=all(chosen["periods"][p]["net_pnl"] > 0
                                                for p in ("2025", "2026_01_09")),
                     later_and_continuous_cost_stress_positive=all(
                         s["net_pnl"] > 0 for s in chosen["doubled_costs"].values()))
    return {"protocol": "docs/evaluation/RISK_BUDGET_RESEARCH_PLAN.md", **selection,
            "scenarios": scenarios, "gates": gates, "research_screen_passed": all(gates.values()),
            "live_eligible": False, "initial_usdt_per_account": 1000., "costs": COSTS,
            "stress_costs": STRESS_COSTS,
            "gap": {"date": str(GAP_DATE), "price_multiplier_from_date": .6,
                    "included_in_historical_screen": False},
            "reconciliation": "all paths: nonnegative cash/inventory, NAV and executed cost totals checked",
            "limitations": ["All source periods, model and present-day survivor universe are already known.",
                "Summed daily highs/lows are conservative bounds, not simultaneous portfolio observations.",
                "Risk orders use prior closed returns, prior high bounds and current open only.",
                "Daily execution, price gaps and minimum notionals can breach an exposure or drawdown target.",
                "Budget floor is never reset; falling below it can prevent future recovery in cash.",
                "Terminal holdings are marked to market; no forced sale or final sale costs.",
                "Latency, liquidity, taxes and cash interest are not modeled."]}


def write_report(report, out):
    out.mkdir(parents=True, exist_ok=True)
    stem = "THETABOT_RISK_BUDGET_2026-10-06"
    (out / f"{stem}.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    pct = lambda x: f"{100*x:+.2f}%"
    frozen = report["frozen_candidate"]
    rows = ["# ThetaBot: výnos při rozpočtu propadu 20 %", "",
        "Uživatel zvýšil přijatelný dočasný propad účtu z 10 % na **20 %**. "
        "Pevné srovnání přidává škálování podle kovariance a podle zbývajícího rozpočtu ztráty. "
        "Momentum signály jsou stejné jako v předchozím výzkumu; nepřidáváme páku.", "",
        f"**Zmrazený výběr z let 2022–2024: `{frozen}`. Historický screening: "
        f"{'prošel' if report['research_screen_passed'] else 'neprošel'}.** "
        "Výběr maximalizuje čistý výnos z vývoje pod 20% konzervativní mezí propadu. "
        "Všechna období už byla známá, proto nejde o nové nezávislé potvrzení výhody.", "",
        "## Výsledky po nákladech", "",
        "| Varianta | Výnos 2025 | Výnos leden–září 2026 | Souvislý výnos 2022–2026 | Souvislý propad ze závěrů | Konzervativní intradenní mez propadu | Všechny historické propady do 20 % |",
        "|---|---:|---:|---:|---:|---:|:---:|"]
    for name, scenario in report["scenarios"].items():
        c = scenario["periods"]["continuous"]
        rows.append(f"| {name}{' (zmrazeno)' if name == frozen else ''} | "
            f"{pct(scenario['periods']['2025']['total_return'])} | "
            f"{pct(scenario['periods']['2026_01_09']['total_return'])} | {pct(c['total_return'])} | "
            f"{pct(c['maxDD'])} | {pct(c['max_intraday_drawdown_bound'])} | "
            f"{'ano' if scenario['all_historical_drawdowns_within_20pct_bound'] else 'ne'} |")
    rows += ["", "Roční přehrávky začínají každá s 1 000 USDT; souvislá přehrávka "
        "používá jediný účet s počátečními 1 000 USDT a nikdy neresetuje hotovost ani rizikové maximum. "
        "Intradenní mez kombinuje denní maxima a minima držených aktiv; nemusely nastat zároveň "
        "ani v pořadí maximum–minimum. Jde o konzervativní mez, nikoli přesně naměřený propad.", "",
        "## Souvislý účet a nákladový stres", "",
        "| Varianta | Stav z 1 000 USDT | Historický roční přepočet | Průměrná expozice | Výnos při dvojnásobných nákladech | Propadová mez při dvojnásobných nákladech |",
        "|---|---:|---:|---:|---:|---:|"]
    for name, scenario in report["scenarios"].items():
        c, stress = scenario["periods"]["continuous"], scenario["doubled_costs"]["continuous"]
        rows.append(f"| {name} | {c['final_equity']:.2f} USDT | {pct(c['cagr'])} | "
            f"{100*c['mean_exposure']:.2f}% | {pct(stress['total_return'])} | "
            f"{pct(stress['max_intraday_drawdown_bound'])} |")
    rows += ["", "Roční přepočet popisuje jednu historickou cestu; není předpovědí výnosu. "
        "Běžný simulovaný náklad je poplatek 0,1 % při každém provedení, skluz 5 bazických bodů "
        "a spread 2 bazické body (polovina při každém provedení). Stres zdvojnásobuje poplatek a skluz. "
        "Náklady jsou účtované za skutečné obchody se společnou hotovostí, nikoli odečtené paušálně.", "",
        "## Zmrazený výběr a kontrolní podmínky", "",
        "| Podmínka | Výsledek |", "|---|:---:|"]
    for name, passed in report["gates"].items():
        rows.append(f"| {name} | {'splněno' if passed else 'nesplněno'} |")
    if frozen != "cash":
        chosen = report["scenarios"][frozen]
        rows += ["", f"Vývojový čistý výnos `{frozen}`: {pct(chosen['development']['total_return'])}; "
            f"konzervativní mez propadu {pct(chosen['development']['max_intraday_drawdown_bound'])}. "
            "Volba byla zapsaná před výpočtem novějších období; jiného pozdějšího vítěze automaticky nenasazujeme.", "",
            f"Nákladový stres vybrané varianty: 2025 {pct(chosen['doubled_costs']['2025']['total_return'])}, "
            f"leden–září 2026 {pct(chosen['doubled_costs']['2026_01_09']['total_return'])}."]
    rows += ["", "## Umělý společný cenový skok −40 %", "",
        "Předem zvolený scénář násobí ceny všech tří aktiv od otevření 11. března 2024 faktorem 0,6. "
        "Následující relativní pohyby zachovává, signály i řízení znovu počítá. "
        "Jde o zátěžovou konstrukci, nikoli další pozorovaný historický výsledek. "
        "Test ukazuje, zda otevřený cenový skok může porušit 20% rozpočet ještě před prodejem.", "",
        "| Varianta | Souvislý výnos ve scénáři | Propad ze závěrů | Konzervativní mez propadu |",
        "|---|---:|---:|---:|"]
    for name, scenario in report["scenarios"].items():
        gap = scenario["correlated_40pct_gap"]
        rows.append(f"| {name} | {pct(gap['total_return'])} | {pct(gap['maxDD'])} | "
                    f"{pct(gap['max_intraday_drawdown_bound'])} |")
    rows += ["", "## Pravidla a omezení", "",
        "`vol_15` a `vol_25` používají 60 uzavřených denních výnosů a kovarianci "
        "staženou o 25 % k diagonále; cíle jsou 15 % a 25 % roční volatility portfolia. "
        "`cushion_3` a `cushion_5` omezují investovanou část na 3× nebo 5× zbývající odstup "
        "od podlahy 82 % historického maxima účtu. Kombinace respektuje oba limity. "
        "Rizikové maximum zahrnuje konzervativní cenovou mez předchozích uzavřených dní. "
        "Aktuální denní rozsah ceny nesmí ovlivnit příkazy na jeho otevření.", "",
        "Běžné nákupy a přesuny jsou v pondělí s pásmem 1 % NAV. Výstup neaktivního signálu "
        "a rizikové snížení probíhají na nejbližším denním otevření; rizikové snížení obchází toto pásmo. "
        "Minimum 10 USDT zůstává platné a může ponechat malý zbytek pozice. "
        "Podlaha ani maximum se po ztrátě neresetují; účet pod podlahou může zůstat v hotovosti "
        "bez možnosti obnovit výnos. Četnější rizikové redukce mohou zvýšit náklady.", "",
        "20% hranice je kritérium výběru. Denní řízení ani prodejní příkaz nezaručí její "
        "dodržení při budoucím skoku ceny, omezené likviditě nebo zpoždění. "
        "Žádná varianta není tímto výpočtem zapnutá pro živé obchodování. "
        "Otevřené konečné pozice jsou oceněné trhem; konečná nucená likvidace a její náklady, "
        "úročení hotovosti, daně a skutečná likvidita nejsou modelované.", "",
        "Všechny uložené cesty byly účetně zkontrolovány: nezáporná hotovost a pozice, "
        "NAV ze skutečného ocenění pozic, poplatky a cenový dopad souhlasí s obchody. "
        "To ověřuje provedení simulace, nikoli budoucí výdělečnost.", "",
        "[Předem zapsaný protokol](RISK_BUDGET_RESEARCH_PLAN.md); doprovodný JSON obsahuje "
        "metriky všech přehrávek, rizikové nastavení, kontrolní podmínky a zdrojové manifesty.", "",
        "Inspirace: [Moreira a Muir, Volatility Managed Portfolios](https://www.nber.org/papers/w22208). "
        "Omezení provedení stop příkazů: [SEC / Investor.gov](https://www.investor.gov/introduction-investing/general-resources/news-alerts/alerts-bulletins/investor-bulletins-14). "
        "Tyto zdroje nepotvrzují výdělečnost tohoto krypto modelu.", "",
        "## Opakování", "", "```bash",
        "python -m scripts.research_risk_budget --daily data/raw/*_1d_*.csv --report-out docs/evaluation",
        "```", ""]
    (out / f"{stem}.md").write_text("\n".join(rows))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts/risk_budget"))
    parser.add_argument("--report-out", type=Path, default=None)
    parser.add_argument("--as-of", default=None)
    args = parser.parse_args(argv)
    as_of = pd.to_datetime(args.as_of, utc=True) if args.as_of else pd.Timestamp.now(tz="UTC")
    markets, provenance = load_daily(args.daily, as_of)
    report = {**research(markets, args.out), "as_of": str(as_of), "daily_provenance": provenance}
    (args.out / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if args.report_out is not None:
        write_report(report, args.report_out)
    print(f"Complete: frozen={report['frozen_candidate']}; screen={report['research_screen_passed']}")


if __name__ == "__main__":
    main()
