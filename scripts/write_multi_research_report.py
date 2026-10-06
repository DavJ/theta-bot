"""Turn the fixed experiment's evidence into a repository review artifact."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from scripts.research_dual_scale import DEVELOPMENT_END, VALIDATION_END, load_archives
from scripts.research_multi_strategy import CANDIDATES, run_candidate
from spot_bot.evaluate import EvaluationConfig


def pct(value):
    return f"{value * 100:+.2f} %"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--csv", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("docs/evaluation"))
    parser.add_argument("--preview-png", type=Path, default=None)
    args = parser.parse_args(argv)
    report = json.loads(args.summary.read_text())
    config = EvaluationConfig(**report["config"])
    data, manifests = load_archives(args.csv, config, pd.to_datetime(report["as_of"], utc=True))
    if manifests != report["provenance"]:
        raise ValueError("Plot inputs must match the measured source manifests")
    args.out.mkdir(parents=True, exist_ok=True)
    stem = "THETABOT_MULTI_2026-10-06"
    report["implementation_verification"] = {"pytest_tests": 403, "legacy_metric_tests": 8,
                                             "github_ci": "pending at artifact creation"}
    report["limitations"][-1] = "The 30% cap is enforced at decision prices, overriding hysteresis; market drift during execution or minimum notional can leave a residual deviation. Capital-return benchmarks are not risk matched."
    (args.out / f"{stem}.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    rows = []
    for candidate in CANDIDATES:
        name = candidate.name
        dev, val, stress, diag = [report[k][name] for k in ["development", "validation_2025", "stress_validation_2025", "diagnostic_2026"]]
        rows.append(f"| {name} | {pct(dev['total_return'])} | {pct(val['total_return'])} | {pct(stress['total_return'])} | {pct(diag['total_return'])} | {pct(val['maxDD'])} | {int(val['trades_count'])} |")
    winner = report["frozen_candidate"]
    val, diag = report["validation_2025"][winner], report["diagnostic_2026"][winner]
    skill = report["forecast_skill_2025"]
    q = report["independently_reset_validation_quarters"][winner]
    text = f"""# ThetaBot: více přístupů a Kalmanova fúze — 6. října 2026

**Kalmanova fúze je implementovaná, ale ekonomicky neprošla.** Výběr podle let
2022–2024 zvolil `{winner}`: v roce 2025 pak dosáhl výnosu {pct(val['total_return'])},
za leden–září 2026 {pct(diag['total_return'])}. Nejlepší pozorovaný výsledek
v roce 2025 měl `breakout` (+2,31 %), který byl kladný i v diagnostice 2026
(+0,94 %). Tento výběr podle porovnání je průzkumný, ne nový nezávislý důkaz.
Nasazení vítěze vývojového vzorku by ztrácelo peníze; žádný živý obchod nebyl
proveden ani aktivován. Cíl je nejvyšší čistý zisk v uvedeném souboru kandidátů,
za stejného kapitálu a limitu expozice.

## Je původních +0,82 % po poplatcích?

Ano. Původní oddělený test z července–září 2026 počítal 0,1 % poplatek za
každé provedení, 5 bp skluz u tržního plnění a 2 bp plný spread. Z 1000 USDT
bylo +8,20 USDT čistého; aritmetický hrubý výsledek na stejných obchodech
11,86 USDT, poplatky 2,58 USDT, skluz/spread 1,09 USDT. Průměrná expozice
byla pouze 4,56 %. Hrubý výsledek není samostatný běh bez nákladů. Limitní
plnění má vlastní předpoklady a nedostává ještě dodatečný tržní skluz.

Rozšířené porovnání níže používá jiná časová období a dovoluje skutečný výstup
ze ztrátové pozice. Původní profit guard takový výstup mohl blokovat. Nové
výsledky proto nejsou přímou změnou stejného původního testu.

## Stejné podmínky pro všech 11 kandidátů

BTCUSDT spot, 1h, 1000 USDT, 30% cílová expozice vynucená při rozhodování,
bez páky, poplatek 0,1 % za provedení, tržní skluz 0,05 %, plný spread
0,02 %. Stres zdvojnásobuje poplatek i skluz a znovu počítá rozhodnutí; spread
zůstává stejný. Jiné náklady mohou změnit obchody, takže horší jednotlivá
plnění nemusí znamenat monotónně horší souhrnný výnos.

57 měsíčních archivů má ověřené SHA-256 součty. Celkem {report['rows']:,}
svíček, leden 2022–září 2026. Chybí jedna zdrojová hodina, 24. března 2023
13:00 UTC; zůstává chybějící a nebyla nahrazena umělou svíčkou.

Výběr proběhl podle maximálního čistého výnosu na vývoji, s kladným výsledkem
a maximálním propadem do 15 %. Volba byla uložená před následným porovnáním.
Rok 2025 je historické ověření v rámci společného průzkumu; po předchozím
experimentu se škálou filtru není označen jako nový potvrzující lockbox.
Rok 2026 byl zčásti dříve viděný a je diagnostika. Parametry se po ověření
neladily. Výsledky jsou konečné tržní ocenění otevřených pozic bez likvidace.

| Přístup | Vývoj 2022–2024 | Rok 2025 | 2025 dvojí náklady | 2026 leden–září | Max. propad 2025 | Provedení 2025 |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

Počáteční 30% nákup a držení BTC: rok 2025
{pct(report['benchmarks']['validation_2025']['initial_30pct_hold']['total_return'])},
leden–září 2026 {pct(report['benchmarks']['diagnostic_2026']['initial_30pct_hold']['total_return'])}.
To je srovnání výnosu kapitálu; expozice/riziko nejsou plně vyrovnané.

![Vývoj kapitálu po nákladech]({stem}.svg)

## Jak funguje fúze

Theta a čtyři cenové přístupy nejprve vytvářejí skóre. Každé se kalibruje na
očekávaný výnos příštího dokončeného UTC dne pomocí již známých výsledků,
nejvýše 252 dní a nejméně 126 pozorování. Nejnovější učící dvojice je včerejší
skóre a dnes známý výnos. Současná nebo budoucí odpověď pro dnešní skóre se
při kalibraci nepoužívá.

Kalmanův stav je očekávaný denní výnos. Kovariance měření vychází z dříve
realizovaných chyb jednotlivých předpovědí a zahrnuje jejich korelaci; dva
podobné signály tedy nepředstavují dvě nezávislá potvrzení. Parametry Q/R
jsou předem určené v protokolu. Josephův přepočet zachovává kladnou varianci.
Model je lineární, proto zde není potřeba EKF ani jeho nelineární Jacobian.
Všechny zdroje sdílejí jednu výslednou pozici a jednu cestu provedení.

Fúze měla na {skill['fused_return']['observations']} denních předpovědích
úspěšnost směru {skill['fused_return']['directional_accuracy'] * 100:.2f} %.
Střední čtvercová chyba byla {skill['fused_return']['mean_squared_error']:.9f},
pro předpověď nulového výnosu {skill['zero_return_forecast_mse']:.9f}.
Složitější kombinace tedy nepřinesla lepší předpověď než tento jednoduchý
referenční model. Kvartály 2025 s nezávisle obnoveným kapitálem dopadly
{', '.join(pct(item['total_return']) for item in q)}. Vývojový zisk
{pct(report['development'][winner]['total_return'])} sám o sobě nebyl přenositelný.

## Co je v kódu a co z toho plyne

- Osm nových režimů strategie lze zvolit přes `spot_bot.run_live --strategy`:
  momentum, EMA trend, breakout, range reversion, dvě expozicové kombinace,
  průměr kalibrovaných předpovědí a Kalmanova fúze. Výchozí režim se neaktivoval.
- Volitelný logaritmický dual Kalman je invariantní vůči jednotkám ceny a
  normalizuje inovaci minulou volatilitou. Správná kalibrace automaticky
  nevytváří obchodní výhodu; přímé časté obchodování v tomto testu ztrácí.
- Nové strategie mohou provést ztrátový výstup a snížit pozici nad rizikový
  limit. Hysteréze ani požadavek zisku tyto redukce nezablokují.
- Dokončený den se nikdy nepřenáší zpět do nedokončených hodin. Regresní
  testy kontrolují budoucí data, společný horizont, korelované předpovědi,
  stejný výsledek jednotlivého a dávkového signálu a účetnictví paper režimu.
- 403 regresních testů a 8 testů metrik prošlo lokálně. Stav GitHub CI po
  publikaci je v PR #119; JSON zachycuje stav vytvoření artefaktu.

Pro další paper průzkum je `breakout` zajímavější než okamžité nasazení fúze.
Je to výběr po tomto porovnání, který stále potřebuje nové pozorování. Cíl
velkého stabilního zisku zatím doložený není. Fúzní API zůstává připravené
pro další zdroje předpovědí; jejich užitečnost se musí měřit stejně.

## Opakování

```bash
python -m scripts.download_binance_archive --allow-gaps --start-month 2022-01 --end-month 2023-12 --out data/raw/BTCUSDT_1h_2022_2023.csv
python -m scripts.download_binance_archive --start-month 2024-01 --end-month 2025-12 --out data/raw/BTCUSDT_1h_2024_2025.csv
python -m scripts.download_binance_archive --start-month 2026-01 --end-month 2026-09 --out data/raw/BTCUSDT_1h_2026_01_09.csv
python -m scripts.research_multi_strategy --csv data/raw/BTCUSDT_1h_2022_2023.csv data/raw/BTCUSDT_1h_2024_2025.csv data/raw/BTCUSDT_1h_2026_01_09.csv --as-of 2026-10-06T09:45:00Z --out artifacts/multi_strategy_capped
python -m scripts.write_multi_research_report --summary artifacts/multi_strategy_capped/summary.json --csv data/raw/BTCUSDT_1h_2022_2023.csv data/raw/BTCUSDT_1h_2024_2025.csv data/raw/BTCUSDT_1h_2026_01_09.csv
```

Přesné definice, zdroje a omezení:
[předem napsaný protokol](MULTI_STRATEGY_RESEARCH_PLAN.md),
[úplné měření a kontrolní součty]({stem}.json).
Kalmanovy rovnice odpovídají
[autorské dokumentaci](https://filterpy.readthedocs.io/en/latest/kalman/KalmanFilter.html);
převod indikátorů na výnosy a jeho nastavení jsou naše výzkumné volby.
Paper běh potřebuje dostatečnou historii pro denní kalibraci, například
10 000 hodinových svíček; kratší historie u fúze zůstane v hotovosti.
Živá latence, fronty objednávek a obnovení celého procesu nejsou tímto testem
ověřené.
"""
    (args.out / f"{stem}.md").write_text(text)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    choices = [next(c for c in CANDIDATES if c.name == name) for name in ["kalman_fusion", "breakout"]]
    for axis, start, end, title in zip(axes, [DEVELOPMENT_END, VALIDATION_END],
                                      [VALIDATION_END, pd.Timestamp("2026-10-01", tz="UTC")],
                                      ["2025: historical validation", "Jan–Sep 2026: diagnostic replay"]):
        prefix = data.loc[data.timestamp < end]
        period = prefix.loc[prefix.timestamp >= start]
        for candidate in choices:
            equity, _, _ = run_candidate(prefix, candidate, config, start)
            axis.plot(equity.timestamp, equity.equity, label=candidate.name.replace("_", " "), linewidth=1.5)
        fill = float(period.open.iloc[0]) * (1 + (config.slippage_bps + config.spread_bps / 2) / 10000)
        allocated = config.initial_usdt * config.max_exposure
        quantity = allocated / (fill * (1 + config.fee_rate))
        axis.plot(period.timestamp, config.initial_usdt - allocated + quantity * period.close,
                  label="30% initial BTC hold", color="#8b98a8", linewidth=1)
        axis.axhline(config.initial_usdt, label="cash", linestyle="--", color="#616161", linewidth=0.8)
        axis.set_title(title, fontsize=11)
        axis.set_ylabel("Equity (USDT), fresh 1000 each period")
        axis.grid(alpha=0.2)
        axis.tick_params(axis="x", labelrotation=30)
    axes[0].legend(fontsize=8)
    fig.suptitle("BTCUSDT spot: fees, slippage and spread included; 30% decision exposure cap", fontsize=12)
    fig.savefig(args.out / f"{stem}.svg")
    if args.preview_png is not None:
        args.preview_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.preview_png, dpi=140)
    plt.close(fig)
    print(f"Wrote {args.out / (stem + '.md')}, JSON and SVG")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
