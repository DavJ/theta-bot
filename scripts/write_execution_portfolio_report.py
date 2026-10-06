"""Render the fixed experiment's complete evidence without changing selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("artifacts/execution_portfolio"))
    parser.add_argument("--out", type=Path, default=Path("docs/evaluation"))
    parser.add_argument("--preview-png", type=Path)
    parser.add_argument("--pytest-count", type=int, default=0)
    args = parser.parse_args(argv)
    result = json.loads((args.source / "summary.json").read_text())
    p = result["portfolio"]
    frozen = p["frozen_candidate"]
    stem = "THETABOT_EXECUTION_PORTFOLIO_2026-10-06"
    args.out.mkdir(parents=True, exist_ok=True)
    result["implementation_verification"] = {"pytest_tests": args.pytest_count,
                                               "legacy_metrics_tests": 8,
                                               "economic_screen_is_separate_from_code_tests": True}
    (args.out / f"{stem}.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    pct = lambda x: f"{100*x:+.2f}%"
    lines = ["# ThetaBot: provedení obchodů a pomalé portfolio, 6. října 2026", "",
             "**Spolehlivě ziskového bota pro živé obchodování tento experiment nepotvrdil.** "
             "Výrazně lepším průzkumným kandidátem je 30denní momentum na BTC, ETH a BNB. "
             "Varianta vybraná pouze podle starších dat ale na novějších obdobích neprošla. "
             "Lepší zpětně pozorované výsledky nejsou novým potvrzením výběru.", "",
             "## Výsledek všech šesti předem určených variant", "",
             "Každé období začíná s novými 1 000 USDT; historie signálů zůstává dostupná. "
             "Rok 2022 je zahřívací historie. Výběr podle čistého zisku proběhl na letech 2023–2024 "
             "při maximálním vývojovém propadu 15 %. Výběr byl zapsán před vyhodnocením 2025 a 2026.", "",
             "| Varianta | Vývoj 2023–24 | Rok 2025 | Leden–září 2026 | Propad 2025 | Propad 2026 |",
             "|---|---:|---:|---:|---:|---:|"]
    names = {"btc_breakout": "Průraz, pouze BTC", "breakout_equal": "Průraz, tři aktiva",
             "momentum_vol": "30denní momentum, váhy podle volatility",
             "horizons_equal": "Čtyři horizonty, stejné základní váhy",
             "horizons_vol": "Čtyři horizonty, váhy podle volatility", "rotation": "90denní rotace"}
    for name, d in p["development"].items():
        v, t = p["validation_2025"][name], p["transfer_2026"][name]
        lines.append(f"| {names[name]} | {pct(d['total_return'])} | {pct(v['total_return'])} | "
                     f"{pct(t['total_return'])} | {pct(v['maxDD'])} | {pct(t['maxDD'])} |")
    lines += ["", f"**Zmrazená volba: `{frozen}`. Ekonomický screening: {p['research_screen_passed']}. "
              "Živé nasazení: ne.**", "", "| Kontrola zmrazené volby | Výsledek |", "|---|---|"]
    for name, passed in p["gates"].items():
        lines.append(f"| {name} | {'prošlo' if passed else 'neprošlo'} |")
    if frozen != "cash":
        lines += ["", "Při dvojnásobném poplatku a skluzu: " + "; ".join(
                  f"{year}: {pct(s['total_return'])}" for year, s in p["doubled_costs_frozen"].items()) + ".",
                  "Nezávisle resetovaná čtvrtletí 2025: " + ", ".join(
                  pct(q["total_return"]) for q in p["independent_2025_quarters"]) + "."]
    lines += ["", "## Náklady lepšího průzkumného kandidáta", "",
              "`momentum_vol` byl jedním z předem určených modelů. Níže jej vyzdvihujeme až po "
              "porovnání novějších výsledků; nejde o zmrazeného vítěze ani prokázanou budoucí výhodu.", "",
              "| Období | Čistý zisk z 1 000 USDT | Poplatky | Skluz a spread | Počet provedení | Průměrná expozice |",
              "|---|---:|---:|---:|---:|---:|"]
    for key, label in [("validation_2025", "2025"), ("transfer_2026", "2026 leden–září")]:
        s = p[key]["momentum_vol"]
        lines.append(f"| {label} | {s['net_pnl']:+.2f} USDT | {s['fees_paid_total']:.2f} USDT | "
                     f"{s['slippage_paid_total']:.2f} USDT | {s['trades_count']} | {100*s['mean_exposure']:.2f}% |")
    lines += ["", "Poplatek je 0,1 % při každém nákupu i prodeji. Každé tržní provedení dále "
              "platí 5 bazických bodů skluzu a polovinu spreadu 2 bazické body. "
              "Obchody se provádějí na následujícím denním otevření ze známého předchozího zavření. "
              "Běžné změny vah probíhají v pondělí, s pásmem 1 % NAV a minimálním obchodem 10 USDT. "
              "Výstupy do hotovosti a snížení překročené expozice jsou denní.", "",
              "Cílová celková expozice je nejvýše 30 %, u portfolia nejvýše 15 % na aktivum. "
              "Pohyb trhu v průběhu dne a minimální velikost obchodů způsobují odchylky. "
              "U momentum kandidáta bylo maximum měřené na denních zavřeních "
              f"{100*p['validation_2025']['momentum_vol']['max_close_exposure']:.2f}% v roce 2025 a "
              f"{100*p['transfer_2026']['momentum_vol']['max_close_exposure']:.2f}% v roce 2026.", "",
              "## Co změnily opravy původního bota", "",
              "Opraven je chybějící druhý skluz v prahu zpátečního obchodu, směšování jednostranných "
              "a zpátečních nákladů a skutečné dodržení minima/maxima hystereze. Denní signál je "
              "dostupný při zavření poslední svíčky dne; už se nezpožďuje o další hodinovou svíčku. "
              "Přidána je explicitní politika `market` pro simulátor a plánovač.", "",
              "Níže jsou nové diagnostické přehrávky původních strategií po těchto opravách. "
              "Limitní režim nabídne limit a po jedné nedotčené svíčce přejde na trh při zavření. "
              "Tržní režim provede příkaz na známém otevření. Obě varianty platí skutečné náklady.", "",
              "| Strategie | 2025 limit/timeout | 2025 tržně | 2026 limit/timeout | 2026 tržně |",
              "|---|---:|---:|---:|---:|"]
    for name in result["execution"]["2025"]:
        a, b = result["execution"]["2025"][name], result["execution"]["2026"][name]
        lines.append(f"| `{name}` | {pct(a['limit_then_market']['total_return'])} | {pct(a['market']['total_return'])} | "
                     f"{pct(b['limit_then_market']['total_return'])} | {pct(b['market']['total_return'])} |")
    lines += ["", "Samotné opravy provedení nestačí: původní theta modely stále intenzivně obchodují "
              "a náklady převyšují jejich slabý přínos. Kalmanova fúze i v tržním režimu zůstala ztrátová. "
              "Počet regresních testů není důkaz ziskovosti.", "", "## Graf a omezení", "",
              f"![Denní čistá hodnota portfolia]({stem}.svg)", "",
              "Počáteční stejně rozdělené 30% držení tří aktiv mělo v roce 2025 "
              f"{pct(p['benchmarks']['2025']['initial_30pct_equal_hold_return'])} a v roce 2026 "
              f"{pct(p['benchmarks']['2026']['initial_30pct_equal_hold_return'])}. "
              "Je to srovnání kapitálového výnosu, nikoli přesně srovnaného rizika.", "",
              "BTC období byla prohlížena dříve a jsou nyní známá výzkumná data. ETH a BNB jsou nové "
              "zdroje pro tento experiment, ale výběr dnešních přeživších aktiv zavádí výběrové zkreslení. "
              "Ani toto porovnání není živý dopředný test. Nejsou modelovány fronty příkazů, "
              "latence nebo nedostatečná likvidita. Konečné pozice nejsou nuceně likvidovány.", "",
              "Celkem 171 měsíčních archivů denních dat, 5 202 skutečných denních svíček, bez doplňování "
              "chybějících cen. Každý archiv byl ověřen oficiálním SHA-256 checksumem. "
              "Úplné manifesty, hash dat, všechny metriky i neúspěšné kontroly jsou v doprovodném JSON.", "",
              f"Ověření implementace: {args.pytest_count} regresních testů a 8 testů metrik. "
              "Kontrolují mimo jiné budoucí neměnnost signálů, jednotné účtování a sdílený účet.", "",
              "## Opakování", "", "```bash",
              "python -m scripts.research_execution_portfolio --daily data/raw/*_1d_*.csv \\",
              "  --hourly-btc data/raw/BTCUSDT_1h_2022_2023.csv data/raw/BTCUSDT_1h_2024_2025.csv \\",
              "  data/raw/BTCUSDT_1h_2026_01_09.csv --out artifacts/execution_portfolio --jobs 3",
              "python -m scripts.write_execution_portfolio_report --source artifacts/execution_portfolio",
              "```", "", "Denní archivy lze znovu získat nástrojem `scripts.download_binance_archive` "
              "s `--timeframe 1d` pro každý ze tří symbolů a bloky 2022–23, 2024–25 a leden–září 2026. "
              "Protokol je v [EXECUTION_PORTFOLIO_RESEARCH_PLAN.md](EXECUTION_PORTFOLIO_RESEARCH_PLAN.md).", "",
              "Hypotézu motivují původní studie [Time Series Momentum](https://www.aqr.com/Insights/Research/Journal-Article/Time-Series-Momentum) "
              "a [Common Risk Factors in Cryptocurrency](https://www.nber.org/papers/w25882). "
              "Toto je jejich jednoduchá spotová adaptace, ne replikace výsledků.", ""]
    (args.out / f"{stem}.md").write_text("\n".join(lines))
    with plt.rc_context({"svg.fonttype": "none", "font.family": "DejaVu Sans", "axes.spines.top": False,
                         "axes.spines.right": False, "path.simplify": True}):
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
        selected = [("momentum_vol", "30denní momentum", "#007f86"),
                    (frozen, "Výběr ze starších dat", "#be663d"),
                    ("btc_breakout", "Průraz pouze BTC, denní režim", "#627191")]
        for ax, year in zip(axes, ["2025", "2026"]):
            for name, label, color in selected:
                if name == "cash":
                    continue
                f = pd.read_csv(args.source / f"{name}_{year}_equity.csv", parse_dates=["timestamp"])
                ax.plot(f.timestamp, f.equity, label=label, color=color, linewidth=1.6)
            ax.axhline(1000, color="#87929b", linestyle="--", linewidth=1, label="Hotovost")
            ax.set_title(year if year == "2025" else "Leden–září 2026")
            ax.grid(alpha=.18)
            ax.tick_params(axis="x", rotation=25)
        axes[0].set_ylabel("Čistá hodnota z počátečních 1 000 USDT")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
        fig.suptitle("Předem určené portfolio: lepší kandidát, neprošlá zmrazená volba", fontsize=14)
        fig.tight_layout(rect=(0, .14, 1, .91))
        fig.savefig(args.out / f"{stem}.svg")
        if args.preview_png:
            fig.savefig(args.preview_png, dpi=160)
        plt.close(fig)
    print(args.out / f"{stem}.md")


if __name__ == "__main__":
    main()
