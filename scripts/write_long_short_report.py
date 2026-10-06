"""Render the fixed derivatives study without changing the frozen selection."""
import json
from pathlib import Path


def write_report(report, out, equity_dir=None):
    out.mkdir(parents=True, exist_ok=True)
    stem = "THETABOT_LONG_SHORT_2026-10-06"
    (out / f"{stem}.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    pct = lambda x: f"{100*x:+.2f}%"
    frozen = report["frozen_candidate"]
    rows = ["# ThetaBot: long/short, relativní obchody a skutečný funding", "",
        "Dosavadní spot momentum přineslo +103,91 % za leden 2022 až září 2026 "
        "(16,18 % ročně) při konzervativní mezi propadu −18,54 %. To nesplňuje "
        "uživatelův cíl vysokého zisku. Toto kolo mění obchodní mechanismus: "
        "přidává shorty, několik trendových horizontů, EMA, průraz, relativní sílu, "
        "regresní páry a jejich pevnou kombinaci. **Výzkumný limit propadu zůstává 20 %.**", "",
        f"**Vývojový výběr 2022–2024: `{frozen}`; úplný historický screening "
        f"{'prošel' if report['historical_screen_passed'] else 'neprošel'}.** "
        "Volba je zapsaná před výpočtem novějších ekonomických výsledků. Novější "
        "období ani jiné pozdější vítěze nepoužíváme k jejímu přepnutí. Celá "
        "krypto historie i dnešní trojice přeživších aktiv už byly známé: toto "
        "není nový nezávislý holdout ani potvrzení budoucího zisku.", ""]
    if frozen == "spot_control":
        rows += ["**Toto kolo nepřineslo potvrzené zlepšení zisku.** Vývojové pravidlo "
            "ponechalo dosavadní spotovou kontrolu; nové varianty nepřepínáme do bota "
            "jen podle příznivého jednotlivého období.", ""]
    if frozen != "cash":
        c = report["scenarios"][frozen]["periods"]["continuous"]
        control = report["scenarios"]["spot_control"]["periods"]["continuous"]
        rows += [f"Souvislý čistý výnos zmrazené varianty: **{pct(c['total_return'])}**, "
            f"roční přepočet {pct(c['cagr'])}, konzervativní mez propadu "
            f"{pct(c['max_intraday_drawdown_bound'])}. Rozdíl proti spotové kontrole "
            f"je {100*(c['total_return']-control['total_return']):+.2f} procentního bodu.", ""]
    rows += ["## Výnos po poplatcích, cenovém dopadu a fundingu", "",
        "| Varianta | 2025 | Leden–září 2026 | Souvisle 2025–září 2026 | Souvisle 2022–září 2026 | Roční přepočet celé cesty | Konzervativní mez propadu celé cesty |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for name, s in report["scenarios"].items():
        p, c = s["periods"], s["periods"]["continuous"]
        rows.append(f"| {name}{' (zmrazeno)' if name == frozen else ''} | "
            f"{pct(p['2025']['total_return'])} | {pct(p['2026_01_09']['total_return'])} | "
            f"{pct(p['later_continuous']['total_return'])} | {pct(c['total_return'])} | "
            f"{pct(c['cagr'])} | {pct(c['max_intraday_drawdown_bound'])} |")
    rows += ["", "Samostatné roky mají čerstvý účet 1 000 USDT. Obě souvislé cesty "
        "mají jeden účet, žádný reset hotovosti, pozic ani maxima. Délka celé cesty "
        "je 1 734 dní, kratší souvislé cesty 638 dní. Roční přepočet je popis historie, "
        "nikoli odhad budoucího výnosu. Konečné pozice jsou oceněné trhem; náklady "
        "na nucené poslední uzavření nejsou účtované.", "",
        "Spotová kontrola má denní OHLC meze; futures používají hodinové mark OHLC. "
        "Součty extrémů více aktiv nemusely nastat zároveň ani v pořadí maximum–minimum. "
        "Hranice je konzervativní mez, nikoli přesně naměřený intradenní propad. "
        "Kontrola `perp_long_control` používá stejný spotový signál a týdenní rozvrh, "
        "ale jiné provedení, mark ocenění a placený funding; izoluje změnu instrumentu.", ""]
    if equity_dir is not None:
        _plot(report, out / f"{stem}.svg", equity_dir)
        rows += [f"![Nepřerušené účty a meze propadu]({stem}.svg)", ""]
    rows += ["## Vývoj a splnění rizikového rozpočtu", "",
        "| Varianta | Vývojový čistý výnos 2022–2024 | Vývojová mez propadu | Celá cesta: denní závěry | Celá cesta: hodinová otevření/závěry | Všechny meze do 20 % včetně stresu | Všechny kolaterálové kontroly |",
        "|---|---:|---:|---:|---:|:---:|:---:|"]
    for name, s in report["scenarios"].items():
        d, c = s["development"], s["periods"]["continuous"]
        rows.append(f"| {name} | {pct(d['total_return'])} | {pct(d['max_intraday_drawdown_bound'])} | "
            f"{pct(c['maxDD'])} | {pct(c['max_observed_drawdown'])} | "
            f"{'ano' if s['all_historical_drawdowns_within_20pct_bound'] else 'ne'} | "
            f"{'ano' if s['all_collateral_screens_passed'] else 'ne'} |")
    rows += ["", "Spotová pozorovaná metrika v pátém sloupci má pouze denní otevření "
        "a závěry. Požadavek 20 % se vyhodnocuje proti konzervativní mezi na každé "
        "cestě, včetně vývoje i dvojnásobných nákladů. Nižší roční cíl volatility "
        "není zárukou tohoto limitu.", "", "## Dvojnásobný poplatek a skluz", "",
        "| Varianta | 2025 | Leden–září 2026 | Souvisle 2025–2026 | Souvisle 2022–2026 | Mez propadu celé cesty |",
        "|---|---:|---:|---:|---:|---:|"]
    for name, s in report["scenarios"].items():
        d = s["doubled_costs"]
        rows.append(f"| {name} | {pct(d['2025']['total_return'])} | {pct(d['2026_01_09']['total_return'])} | "
            f"{pct(d['later_continuous']['total_return'])} | {pct(d['continuous']['total_return'])} | "
            f"{pct(d['continuous']['max_intraday_drawdown_bound'])} |")
    rows += ["", "Běžný simulovaný poplatek je 0,1 % z každého fillu, skluz 5 bps "
        "a plný spread 2 bps (polovina na fill). Stres zdvojnásobuje poplatek a skluz; "
        "spread a skutečné funding sazby zůstávají stejné. Náklady mění NAV a další "
        "velikosti příkazů, proto se všechny cesty přehrávají znovu.", "",
        "## Náklady a kolaterál celé cesty", "",
        "Hodnoty USDT níže se vztahují k jednomu souvislému účtu s počátečními 1 000 USDT. "
        "Kladný čistý funding znamená zaplacený náklad, záporný inkasovaný příjem.", "",
        "| Varianta | Poplatky USDT | Skluz/spread USDT | Funding zaplacený USDT | Funding přijatý USDT | Čistý funding USDT | Obchody | Průměrná hrubá expozice |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, s in report["scenarios"].items():
        c = s["periods"]["continuous"]
        rows.append(f"| {name} | {c['fees_paid_total']:.2f} | {c['slippage_paid_total']:.2f} | "
            f"{c['funding_paid_total']:.2f} | {c['funding_received_total']:.2f} | "
            f"{c['funding_net_paid_total']:.2f} | {c['trades_count']} | {100*c['mean_exposure']:.2f}% |")
    rows += ["", "| Futures varianta | Nejnižší konzervativní mez kolaterálu USDT | Nejvyšší hodinová hrubá expozice | Hodiny pod 5% margin mezí | Hodiny se zápornou mezí hotovosti |",
        "|---|---:|---:|---:|---:|"]
    for name, s in report["scenarios"].items():
        if name == "spot_control":
            continue
        c = s["periods"]["continuous"]
        rows.append(f"| {name} | {c['minimum_collateral_cash']:.2f} | "
            f"{100*c['max_hourly_close_gross_exposure']:.2f}% | {c['margin_breach_hours']} | {c['negative_collateral_hours']} |")
    rows += ["", "Cílový hrubý notional je nejvýše 100 % NAV a absolutní aktivum 50 %. "
        "Cenové pohyby, náklady, minimum příkazu a denní rebalancování mohou skutečnou "
        "expozici mezi rozhodnutími zvýšit. 5% margin mez je předem zvolený výzkumný "
        "filtr, nikoli skutečný burzovní margin tier, stop-loss nebo simulace likvidace/ADL. "
        "Jakékoli porušení této meze či záporná kolaterálová hotovost variantu vyřazuje.", "",
        "## Zmrazená volba: kontrolní podmínky", "", "| Podmínka | Výsledek |", "|---|:---:|"]
    for name, passed in report["gates"].items():
        rows.append(f"| {name} | {'splněno' if passed else 'nesplněno'} |")
    rows += ["", "## Zdroje a přesnost simulace", "",
        "Futures mají 513 oficiálních měsíčních archivů s ověřeným SHA-256: "
        "5 202 denních obchodních svíček, 124 848 hodinových mark svíček "
        "a 15 606 skutečných funding událostí. Každé aktivum pokrývá všech 1 734 dní. "
        "Časy fundingu včetně milisekundového jitteru a intervaly jsou zachované. "
        "Původní kontrolované spotové závěry slouží jen ke tvorbě signálů a kovariance.", "",
        "Měsíční mark archivy měly 264 chybějících hodin v 11 dnech. Byly doplněné "
        "z odpovídajících oficiálních denních archivů s jejich vlastními kontrolními součty. "
        "Konečný hodinový grid je kompletní. Zdrojové soubory ani sazby se neinterpolují; "
        "seznam doplnění i hashe sestavených CSV jsou v doprovodném JSON.", "",
        "Starší funding API neposkytuje mark cenu konkrétního vypořádání. Náklad proto "
        "oceňujeme konzervativně hodinovým mark high, příjem mark low. Při provedení "
        "ve stejné hodině se zvolí horší platba z původní a nové pozice. Binance "
        "negarantuje přesný okamžik vypořádání poblíž funding timestampu. "
        "Tato aproximace může výnos podhodnotit; nejde o přesný tickový funding. "
        "Souběžné příjmy a platby různých aktiv se v intradenní mezi započítají "
        "i přes možné mezilehlé kolaterálové zůstatky, nikoli pouze jejich čistý součet.", "",
        "Nové denní příkazy využívají předchozí dokončený spotový závěr a kovarianci "
        "z 60 uzavřených denních výnosů, staženou o 25 % k diagonále. Cíl roční "
        "volatility je 20 %, pouze snižuje velikost. Pásmo rebalancování je 1 % NAV, "
        "minimum 10 USDT. Mimo pásmo probíhají výstupy a omezení rizika či stropů. "
        "Shorty mají podepsanou zásobu a průměrnou vstupní cenu; cash se mění "
        "pouze realizovaným P&L, poplatkem a fundingem, NAV přidává unrealized mark P&L.", "",
        "**Všech 81 cest je účetně zkontrolovaných:** pozice proti podepsaným fillům, "
        "kolaterál proti realizaci/poplatkům/fundingu, NAV proti hodinovým mark závěrům "
        "a navíc nezávislá identita podepsaných obchodních cashflow plus konečná hodnota "
        "zásoby. Skluz není podruhé odečítaný od NAV. Regresní testy ověřují účetnictví "
        "long/short, převrácení pozice, funding jitter, časovou kauzalitu a odmítnutí mezer.", "",
        "Simulace neověřuje skutečnou hloubku trhu, latenci, burzovní zaokrouhlení "
        "množství, úročení kolaterálu, daně ani budoucí stálost vztahů. "
        "Nové strategie nejsou zapnuté pro živé obchodování.", "",
        "[Předem zapsaný protokol](LONG_SHORT_RESEARCH_PLAN.md). "
        "Zdroje: [Binance public archives](https://github.com/binance/binance-public-data), "
        "[funding a okamžik vypořádání](https://www.binance.com/en/support/faq/detail/360033525031), "
        "[mark cena](https://www.binance.com/en/support/faq/detail/360033525071), "
        "[historický výzkum krypto momentum](https://www.nber.org/papers/w24877). "
        "Tyto zdroje nepotvrzují výdělečnost zdejší implementace.", "", "## Opakování", "", "```bash",
        "python -m scripts.download_futures_research",
        "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m scripts.research_long_short \\",
        "  --daily data/raw/*_1d_*.csv --report-out docs/evaluation", "```", ""]
    (out / f"{stem}.md").write_text("\n".join(rows))


def _plot(report, path, equity_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    names = ["spot_control", "perp_long_control", "signed_horizons", "relative", "pairs"]
    chosen = report["frozen_candidate"]
    if chosen != "cash" and chosen not in names:
        names.append(chosen)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, constrained_layout=True,
                             gridspec_kw={"height_ratios": [1.6, 1]})
    for name in names:
        data = pd.read_csv(Path(equity_dir) / f"{name}_continuous_equity.csv")
        dates = pd.to_datetime(data.timestamp, utc=True)
        label = name + (" (zmrazeno)" if name == chosen else "")
        axes[0].plot(dates, data.equity, label=label, linewidth=1.3)
        axes[1].plot(dates, 100 * data.intraday_drawdown_bound, linewidth=1.1)
    axes[0].set_title("ThetaBot: souvislé účty po nákladech a fundingu, známá historie")
    axes[0].set_ylabel("NAV (USDT; start 1 000)")
    axes[0].legend(fontsize=8, loc="upper left")
    axes[1].axhline(-20, color="#b82020", linestyle="--", linewidth=1.2)
    axes[1].set_ylabel("Konzervativní mez propadu (%)")
    for ax in axes:
        ax.grid(alpha=.2)
    fig.savefig(path)
    plt.close(fig)
    if Path(path).suffix == ".svg":
        Path(path).write_text("\n".join(line.rstrip() for line in Path(path).read_text().splitlines()) + "\n")
