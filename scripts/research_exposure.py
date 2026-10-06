"""Unlevered position-size sensitivity on the fixed exploratory momentum model."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import pandas as pd

from scripts.research_execution_portfolio import load_daily, START, VALIDATION, TRANSFER, END
from spot_bot.backtest.portfolio import run_portfolio_backtest
from spot_bot.portfolio.trend import TrendPortfolio


CAPS = (.3, .5, .75, 1.)


def write_report(report, out):
    out.mkdir(parents=True, exist_ok=True)
    stem = "THETABOT_EXPOSURE_2026-10-06"
    (out / f"{stem}.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    pct = lambda value: f"{100*value:+.2f}%"
    rows = ["# ThetaBot: vyšší expozice, vyšší výnos i propad", "",
            "Dosavadní 30% strop byl zachovaný výzkumný předpoklad, nikoli limit zadaný uživatelem. "
            "Nyní porovnáváme stejný momentum model při 30%, 50%, 75% a 100% stropu. "
            "Jde o nákup spotových aktiv z vlastní hotovosti. Žádná páka ani půjčka nejsou použity.", "",
            "## Výsledek na již známých obdobích", "",
            "| Strop cílové expozice | Čistý výnos 2025 | Čistý výnos leden–září 2026 | Souvislý propad 2022–2026 |",
            "|---|---:|---:|---:|"]
    for label, s in report["scenarios"].items():
        rows.append(f"| {label.removesuffix('pct')} % | {pct(s['periods']['2025']['total_return'])} | "
                    f"{pct(s['periods']['2026_01_09']['total_return'])} | {pct(s['continuous']['maxDD'])} |")
    rows += ["", "Obě roční přehrávky začínají nezávisle s 1 000 USDT. Souvislý účet "
             "začíná s jedinými 1 000 USDT na začátku roku 2022 a není mezi roky resetován. "
             "Všechny obchody, kompozice portfolia a náklady se znovu počítají; výnosy nejsou násobkem staré tabulky.", "",
             "## Jediný souvislý účet", "",
             "| Strop | Konečný stav z 1 000 USDT | Celkový výnos za celé období | Historický roční přepočet | Průměrná expozice |",
             "|---|---:|---:|---:|---:|"]
    for label, s in report["scenarios"].items():
        c = s["continuous"]
        rows.append(f"| {label.removesuffix('pct')} % | {c['final_equity']:.2f} USDT | {pct(c['total_return'])} | "
                    f"{pct(c['cagr'])} | {100*c['mean_exposure']:.2f}% |")
    rows += ["", "Roční přepočet je geometrický přepočet jedné historické cesty, nikoli předpověď budoucích výnosů.", "",
             "## Nepříznivý rok a delší růstové období", "",
             "| Strop | Čistý výnos 2022 | Propad 2022 | Souhrnný výnos 2023–2024 | Propad 2023–2024 |",
             "|---|---:|---:|---:|---:|"]
    for label, s in report["scenarios"].items():
        bear, dev = s["periods"]["bear_2022"], s["periods"]["2023_2024"]
        rows.append(f"| {label.removesuffix('pct')} % | {pct(bear['total_return'])} | {pct(bear['maxDD'])} | "
                    f"{pct(dev['total_return'])} | {pct(dev['maxDD'])} |")
    rows += ["", "Rok 2022 začíná bez historie před rokem 2022; prvních 60 dní model čeká na "
             "volatilitu. Nejde tedy o plně zahřátý test od prvního lednového dne. Výnos 2023–2024 "
             "je souhrn za dva roky, nikoli roční výnos.", "", "## Dvojnásobné náklady", "",
             "| Strop | Čistý výnos 2025 | Čistý výnos leden–září 2026 |", "|---|---:|---:|"]
    for label, s in report["scenarios"].items():
        rows.append(f"| {label.removesuffix('pct')} % | {pct(s['doubled_costs']['2025']['total_return'])} | "
                    f"{pct(s['doubled_costs']['2026_01_09']['total_return'])} |")
    rows += ["", "Běžné náklady: 0,1 % poplatek při každém provedení, 5 bazických bodů skluzu "
             "a polovina spreadu 2 bazické body. Nákladový stres zdvojnásobuje poplatek a skluz, spread nechává stejný. "
             "Minimální obchod je 10 USDT, běžné změny probíhají v pondělí s pásmem 1 % NAV. "
             "Výstupy do hotovosti a snížení překročené expozice se kontrolují denně. "
             "Limit na jedno aktivum je polovina celkového stropu.", "",
             "## Co tento výsledek znamená", "",
             "Vyšší kapitálová expozice zvedla výnos i hloubku propadů. Nevytvořila novou "
             "predikční výhodu. Momentum bylo vyzdviženo až po předchozím srovnání novějších dat, "
             "takže tyto výsledky nejsou nezávislým potvrzením jeho výběru. Všechna období jsou známá výzkumná data.", "",
             "Dosavadní 15% limit vývojového propadu byl výzkumný předpoklad, nikoli známá "
             "uživatelova tolerance rizika. Tento experiment žádný limit automaticky neuvolňuje, "
             "nevybírá vítěze a nemění konfiguraci živého obchodování. Historický propad není "
             "záruka maximální budoucí ztráty.", "",
             "Denní simulace vynechává latenci, fronty příkazů a intradenní průběh ceny. "
             "Konečné pozice jsou oceněny trhem, ne nuceně zlikvidovány. Dnešní výběr přeživších "
             "BTC/ETH/BNB nezaručuje nezaujatý výběr aktiv. Příjmy z hotovosti a daně nejsou modelovány.", "",
             "Úplné náklady, obchody, metriky a manifesty oficiálních ověřených archivů jsou v doprovodném JSON. "
             "Protokol: [EXPOSURE_SENSITIVITY_PLAN.md](EXPOSURE_SENSITIVITY_PLAN.md).", "",
             "## Opakování", "", "```bash",
             "python -m scripts.research_exposure --daily data/raw/*_1d_*.csv --report-out docs/evaluation",
             "```", ""]
    (out / f"{stem}.md").write_text("\n".join(rows))


def research(markets, out):
    periods = {"bear_2022": (pd.Timestamp("2022-01-01", tz="UTC"), START),
               "2023_2024": (START, VALIDATION), "2025": (VALIDATION, TRANSFER),
               "2026_01_09": (TRANSFER, END)}
    scenarios = {}
    for cap in CAPS:
        policy = TrendPortfolio("momentum_vol", max_exposure=cap, asset_cap=cap / 2)
        label = f"{int(cap*100)}pct"
        scenario = {"policy": asdict(policy), "periods": {}, "doubled_costs": {}}
        for name, (start, end) in periods.items():
            equity, trades, summary = run_portfolio_backtest(markets, policy, start=start, end=end)
            scenario["periods"][name] = summary
            equity.to_csv(out / f"{label}_{name}_equity.csv", index=False)
            trades.to_csv(out / f"{label}_{name}_trades.csv", index=False)
            print(f"{label} {name}: net={summary['total_return']:.4%}, DD={summary['maxDD']:.4%}", flush=True)
        for name in ["2025", "2026_01_09"]:
            start, end = periods[name]
            _, _, scenario["doubled_costs"][name] = run_portfolio_backtest(
                markets, policy, start=start, end=end, fee_rate=.002, slippage_bps=10, spread_bps=2)
        equity, trades, scenario["continuous"] = run_portfolio_backtest(
            markets, policy, start=periods["bear_2022"][0], end=END)
        years = (equity.timestamp.iloc[-1] - equity.timestamp.iloc[0] + pd.Timedelta("1D")).days / 365
        scenario["continuous"]["cagr"] = (scenario["continuous"]["final_equity"] / 1000) ** (1 / years) - 1
        equity.to_csv(out / f"{label}_continuous_equity.csv", index=False)
        trades.to_csv(out / f"{label}_continuous_trades.csv", index=False)
        print(f"{label} continuous: net={scenario['continuous']['total_return']:.4%}, "
              f"DD={scenario['continuous']['maxDD']:.4%}", flush=True)
        scenarios[label] = scenario
    return {"protocol": "docs/evaluation/EXPOSURE_SENSITIVITY_PLAN.md",
            "scenarios": scenarios, "initial_usdt_per_period": 1000.,
            "costs": {"fee_rate": .001, "slippage_bps": 5, "spread_bps": 2},
            "development_drawdown_reference": .15, "live_eligible": False,
            "selection_winner": None,
            "limitations": ["Sizing sensitivity on already observed prices is not fresh alpha confirmation.",
                            "2022 begins without pre-2022 history and stays in cash during indicator warmup.",
                            "No borrowing, leverage, funding charges or forced-liquidation model.",
                            "Decision-price caps can drift intraday and leave minimum-notional residuals.",
                            "Terminal holdings are marked to market, not liquidated.",
                            "The universe is today's known surviving three-asset selection."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts/exposure_sensitivity"))
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--report-out", type=Path)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    as_of = pd.to_datetime(args.as_of, utc=True) if args.as_of else pd.Timestamp.now(tz="UTC")
    markets, provenance = load_daily(args.daily, as_of)
    report = research(markets, args.out)
    report.update(as_of=str(as_of), provenance=provenance)
    (args.out / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if args.report_out is not None:
        write_report(report, args.report_out)
    print(args.out / "summary.json")


if __name__ == "__main__":
    main()
