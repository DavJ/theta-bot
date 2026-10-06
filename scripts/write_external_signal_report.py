"""Write inspectable external-signal evidence without changing model selection."""
from __future__ import annotations

import json
from pathlib import Path


def write_report(report, out, equity_dir=None):
    out.mkdir(parents=True, exist_ok=True)
    stem = "THETABOT_EXTERNAL_SIGNALS_2026-10-06"
    (out / f"{stem}.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    pct = lambda x: f"{100*x:+.2f}%"
    frozen = report["frozen_candidate"]
    rows = ["# ThetaBot: externí signály a pětidenní predikce", "",
        "Porovnání posuzuje varianty proti **20% výzkumnému limitu propadu účtu** a přidává pět dní dopředu "
        "předpovídaný výnos, průběžně přeučovanou regresi, mělký boosting a kombinaci s momentum. "
        "Externí skupiny jsou přidávané odděleně: forex/akcie/volatilita/sazby/ropa, "
        "sentimentový index a funding/tok agresivních příkazů.", "",
        f"**Výběr zmrazený podle vývoje 2022–2024: `{frozen}`. "
        f"Historický screening {'prošel' if report['historical_screen_passed'] else 'neprošel'}.** "
        "Volba se po novějších výsledcích nemění. Krypto ceny, dřívější momentum volba i "
        "současný výběr tří přeživších aktiv už byly známé. Nové zdroje jsou dnešní historické "
        "snapshoty bez ověřených prvních zveřejnění: výsledky zůstávají výzkumnou diagnostikou.", ""]
    if frozen == "control_momentum":
        rows += ["**Výsledkem tohoto kola není zlepšení strategie:** výběr zůstal u dosavadního "
            "momentum. Žádná přidaná skupina neprokázala zlepšení predikční MSE v předem "
            "stanoveném testu. To je závěr o těchto konkrétních modelech a datech, "
            "nikoli tvrzení, že externí zdroje nikdy nemohou být užitečné.", ""]
    rows += ["## Výnos po simulovaných nákladech", "",
        "| Varianta | 2025 | Leden–září 2026 | Souvislý účet 2022–2026 | Roční přepočet souvislé cesty | Konzervativní mez propadu | Všechny historické propady do 20 % |",
        "|---|---:|---:|---:|---:|---:|:---:|"]
    for name, s in report["scenarios"].items():
        c = s["periods"]["continuous"]
        rows.append(f"| {name}{' (zmrazeno)' if name == frozen else ''} | "
            f"{pct(s['periods']['2025']['total_return'])} | {pct(s['periods']['2026_01_09']['total_return'])} | "
            f"{pct(c['total_return'])} | {pct(c['cagr'])} | {pct(c['max_intraday_drawdown_bound'])} | "
            f"{'ano' if s['all_historical_drawdowns_within_20pct_bound'] else 'ne'} |")
    rows += ["", "Roční tabulky používají nezávislé starty z 1 000 USDT. Souvislá cesta "
        "má jediný účet od začátku roku 2022 a neresetuje hotovost. Roční přepočet není "
        "předpověď. Intradenní mez sčítá maxima/minima držených aktiv, které nemusely nastat "
        "zároveň ani ve stejném pořadí; jde o konzervativní mez, nikoli přesně pozorovaný propad.", "",
        "Zkrácení predikce na pět dní samo o sobě nezaručí rychlejší růst kapitálu. "
        "Cílem je zlepšit rozhodnutí a čistý výnos při stejném rozpočtu rizika; běžné přesuny "
        "portfolia zůstávají týdenní a výstupy či riziková snížení denní.", "",
        "Sentimentový zdroj: **[Alternative.me Fear & Greed](https://alternative.me/crypto/fear-and-greed-index/)**. "
        "Jde o jejich Bitcoinový složený index, částečně postavený na ceně, objemu a volatilitě. "
        "Není to nezávislý NLP rozbor textu zpráv."]
    if equity_dir is not None:
        _plot(report, out / f"{stem}.svg", equity_dir)
        rows += ["", f"![Souvislé účty a konzervativní propadové meze]({stem}.svg)", "",
            "Graf předem stanovených cenových a úplných modelů ukazuje účinek algoritmu "
            "i externích dat v jedné nepřerušované cestě. Čárkovaná červená čára je 20% výzkumná hranice."]
    rows += ["", "## Přidávají zdroje predikční informaci?", "",
        "Stejný algoritmus se porovnává s cenovou variantou na společných dostupných datech "
        "roku 2025 až září 2026. Kladný rozdíl MSE znamená menší chybu po přidání skupiny. "
        "Interval vychází z 20denních bloků a z denního průměru ztráty napříč aktivy, "
        "takže nezapočítává silně korelovaná aktiva a překrývající se cíle jako nezávislé vzorky. "
        "99% interval odpovídá přibližné korekci na pět porovnávaných přídavků.", "",
        "| Přidaná skupina | Cenová reference | Společné dny | Průměrné zlepšení MSE | Blokový 99% interval | Interval celý nad nulou |",
        "|---|---|---:|---:|---|:---:|"]
    for name, d in report["forecast_diagnostics"]["incremental_source_value"].items():
        ci = d.get("bonferroni_99pct_ci")
        interval = f"[{ci[0]:+.3g}, {ci[1]:+.3g}]" if ci else "málo dat"
        delta = f"{d['mean_mse_improvement']:+.3g}" if "mean_mse_improvement" in d else "—"
        rows.append(f"| {name} | {d['reference']} | {d['days']} | {delta} | {interval} | "
                    f"{'ano' if d['significant_99pct'] else 'ne'} |")
    rows += ["", "Nejde o důkaz ekonomické výhody. Výsledek blokového bootstrapu "
        "je přibližný a závisí na zvolené historii i stálosti trhu. Současná korelace "
        "s cenou ani snížení chyby predikce nezaručují obchodovatelný výnos po nákladech.", "",
        "| Model | Párové predikce aktiv | MSE | MSE nulové predikce | Spearman IC |",
        "|---|---:|---:|---:|---:|"]
    for name, m in report["forecast_diagnostics"]["models"].items():
        ic = f"{m['spearman_ic']:+.3f}" if m["spearman_ic"] is not None else "—"
        rows.append(f"| {name} | {m['paired_asset_dates']} | {m['mse']:.6g} | {m['zero_forecast_mse']:.6g} | {ic} |")
    rows += ["", "## Dvojnásobné náklady", "",
        "| Varianta | 2025 | Leden–září 2026 | Souvislý výnos | Souvislá mez propadu |",
        "|---|---:|---:|---:|---:|"]
    for name, s in report["scenarios"].items():
        d = s["doubled_costs"]
        rows.append(f"| {name} | {pct(d['2025']['total_return'])} | {pct(d['2026_01_09']['total_return'])} | "
                    f"{pct(d['continuous']['total_return'])} | {pct(d['continuous']['max_intraday_drawdown_bound'])} |")
    rows += ["", "Běžný poplatek je 0,1 % při každém provedení, skluz 5 bps a plný spread 2 bps "
        "(jeho polovina při každém provedení). Stres zdvojnásobuje poplatek a skluz. "
        "Každý účet se znovu přehrává; jde o skutečné náklady realizovaných simulovaných příkazů.", "",
        "## Dostupnost a kvalita zdrojů", "",
        "| Řada | Záznamy zdroje | Dostupné rozhodovací dny | Dny s úplnými rysy |",
        "|---|---:|---:|---:|"]
    for code, p in report["source_coverage"].items():
        rows.append(f"| {code} | {p['observations']} | {p['available_decisions']}/{p['decisions']} | "
                    f"{p['complete_feature_decisions']} |")
    rows += ["", "ECB EUR/USD a USD/JPY: dostupnost v 18:00 UTC v den reference. "
        "NASDAQ, VIX a desetiletý výnos: dva americké federální pracovní dny; WTI: deset "
        "kalendářních dní. Index Alternative.me: jeden plný den po jeho timestampu. "
        "Spotový tok: po dokončení denní svíčky. Funding: po posledním započteném skutečném "
        "vypořádání plus pět minut. Join bere jen již dostupné záznamy a staré publikace "
        "expirují. Neznámá data se nedoplňují budoucí hodnotou.", "",
        "**Vysoká datová nejistota: chybí první historická zveřejnění a původní verze "
        "FRED/ECB/Alternative dat.** Konzervativní zpoždění tuto nejistotu neodstraňuje. "
        "Časový join a prefixové testy ověřují naše zpracování snapshotu, nikoli historii "
        "všech oprav provedených poskytovatelem. Případné pozdější přínosy těchto řad "
        "proto potřebují potvrzení na archivovaných verzích nebo budoucích datech.", "",
        "**Energetická hypotéza nebyla vyvrácena.** Energetickým vstupem v tomto kole "
        "byla ropa WTI uvnitř celé makro skupiny, nikoli samostatně testovaná cena "
        "elektřiny těžařů. [Cambridge mining report](https://www.jbs.cam.ac.uk/faculty-research/centres/alternative-finance/publications/cambridge-digital-mining-industry-report/) "
        "uvádí, že elektřina tvoří přes 80 % peněžních provozních nákladů dotázaných "
        "těžařů BTC; to dokládá nákladovou vazbu, nikoli automaticky předstih vůči ceně BTC. "
        "[EIA](https://www.eia.gov/electricity/wholesale/) popisuje i ERCOT North, "
        "ale v přímo stažených tabulkách elektřiny pro roky 2022 a 2026 tento hub "
        "**chybí**. Kontrola obsahu je v [záznamu zdrojové kontroly](THETABOT_ENERGY_SOURCE_CHECK_2026-10-06.json). "
        "[Přímý ERCOT report](https://www.ercot.com/mp/data-products/data-product-details?id=NP4-180-ER) "
        "uvádí historické ceny hubů s týdenní aktualizací. Časy historické veřejné "
        "publikace a původní verze se musí zvlášť ověřit: "
        "datum obchodu ani dodávky se nesmí zaměnit za čas dostupnosti tohoto datasetu. "
        "Tyto řady zatím nejsou zahrnuté do výsledků výše. Pro BTC je vhodnější samostatný "
        "energetický test; nelze přenášet stejný těžební mechanismus na celé portfolio. "
        "[Ethereum](https://ethereum.org/developers/docs/consensus-mechanisms/pos/pos-vs-pow) "
        "používá proof of stake s podstatně menší spotřebou elektřiny.", "",
        "Binance archivní zdroje mají ověřené zveřejněné SHA-256 kontrolní součty. "
        "Nově stažené denní ceny a objemy pro tok příkazů se shodují s původními cenami "
        "backtestu. Funding sazby jsou jen signál: spotový účet je neplatí ani neinkasuje.", "",
        "Skutečný historický NLP sentiment zpráv nebyl nahrazen indexem ani vymyšlen. "
        "Veřejná GDELT DOC historie nepokrývá celé trénovací období; první zkušební požadavek "
        "byl navíc omezen HTTP 429. ETF toky, open interest, on-chain likvidita a deduplikovaný "
        "zprávový korpus s prvním časem zachycení zůstávají dalšími hypotézami.", "",
        "## Podmínky zmrazené varianty", "", "| Podmínka | Výsledek |", "|---|:---:|"]
    for name, passed in report["gates"].items():
        rows.append(f"| {name} | {'splněno' if passed else 'nesplněno'} |")
    if report["publication_delay_sensitivity_applicable"]:
        rows += ["", "Při dodatečném třídenním zpoždění všech externích dat:", "",
                 "| Období | Čistý výnos | Mez propadu |", "|---|---:|---:|"]
        for name, s in report["publication_delay_sensitivity"].items():
            rows.append(f"| {name} | {pct(s['total_return'])} | {pct(s['max_intraday_drawdown_bound'])} |")
    else:
        rows += ["", "Zmrazená varianta nepoužívá externí data, proto test dodatečné "
                 "publikační latence neovlivňuje její rozhodnutí."]
    rows += ["", "## Algoritmus a omezení", "",
        "Regrese používá Ridge alpha=10 a scaler fitovaný jen na tréninku. Boosting má "
        "hloubku dvě, minimálně 30 vzorků v listu a pevné parametry. Každých dvacet dní "
        "se fituje posledních nejvýše 504 a nejméně 126 dokončených výsledků. "
        "Pět budoucích denních závěrů cíle musí být již známých. Klipování cíle na ±3 "
        "směrodatné odchylky používá jen trénink. Stávající dual Kalman/theta režimové "
        "rysy zůstávají součástí cenových vstupů.", "",
        "Proměnlivé zpoždění reakce trhu je oddělené od dostupnosti publikace: každý "
        "externí rys má tři zpětná kalendářní okna 0–2, 3–9 a 10–20 dní. Jejich váhy "
        "se mohou měnit při přeučení; boosting může reagovat i na stav trhu. Trénink "
        "má exponenciální váhy s poločasem 126 dní, normalizované na průměr jedna. "
        "Vážený je scaler i regrese/boosting. Jde o hrubý model rozprostřené reakce, "
        "nikoli určení přesného zpoždění události či důkaz příčinnosti. Další Kalman "
        "ani EKF zde zatím nefiltruje váhy.", "",
        "Modelové alokace mají nejvýše 100 % cílové expozice a 50 % na aktivum, "
        "škálované na cíl roční volatility 20 % podle celé kovariance. To není 20% "
        "zaručený maximální propad. Kontrola momentum zachovává 50%/25% strop. "
        "Týdenní pásmo 1 % NAV a minimum 10 USDT mohou ponechat zbytek pozice. "
        "Konečné pozice jsou oceněné trhem bez nucené likvidace. Likvidita, latence, "
        "daně a úročení hotovosti nejsou modelované. Živý bot se tímto experimentem nezapíná.", "",
        "Všechny cesty mají zkontrolovanou nezápornou hotovost a pozice, NAV ze skutečných "
        "cen a náklady podle obchodů. Měsíční výsledky všech účtů a kompletní manifesty "
        "jsou v doprovodném JSON. [Pevný protokol](EXTERNAL_SIGNAL_RESEARCH_PLAN.md).", "",
        "Zdroje: [ECB reference](https://www.ecb.europa.eu/stats/policy_and_exchange_rates/euro_reference_exchange_rates/html/index.en.html), "
        "[NASDAQ / FRED](https://fred.stlouisfed.org/series/NASDAQCOM), "
        "[CBOE VIX / FRED](https://fred.stlouisfed.org/series/VIXCLS), "
        "[US Treasury yield / FRED](https://fred.stlouisfed.org/series/DGS10), "
        "[EIA WTI / FRED](https://fred.stlouisfed.org/series/DCOILWTICO), "
        "[Alternative.me Fear & Greed](https://alternative.me/crypto/fear-and-greed-index/), "
        "[Binance](https://github.com/binance/binance-public-data).", "", "## Opakování", "", "```bash",
        "python -m scripts.download_external_signals",
        "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m scripts.research_external_signals \\",
        "  --daily data/raw/*_1d_*.csv --report-out docs/evaluation", "```", ""]
    (out / f"{stem}.md").write_text("\n".join(rows))


def _plot(report, path, equity_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    names = ["control_momentum", "ridge_price", "ridge_all", "boost_all"]
    chosen = report["frozen_candidate"]
    if chosen != "cash" and chosen not in names:
        names.append(chosen)
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 6.5), sharex=True,
                             gridspec_kw={"height_ratios": [1.5, 1]}, constrained_layout=True)
    for name in names:
        data = pd.read_csv(Path(equity_dir) / f"{name}_continuous_equity.csv")
        dates = pd.to_datetime(data.timestamp, utc=True)
        label = name + (" (zmrazeno)" if name == chosen else "")
        axes[0].plot(dates, data.equity, label=label, linewidth=1.4)
        axes[1].plot(dates, 100 * data.intraday_drawdown_bound, linewidth=1.1)
    axes[0].set_title("ThetaBot: jeden souvislý účet po nákladech, známá historie")
    axes[0].set_ylabel("NAV (USDT; start 1 000)")
    axes[0].legend(fontsize=8, loc="upper left")
    axes[1].axhline(-20, color="#b82020", linestyle="--", linewidth=1.2)
    axes[1].set_ylabel("Konzervativní mez propadu (%)")
    for ax in axes:
        ax.grid(alpha=.2)
    fig.savefig(path)
    plt.close(fig)
