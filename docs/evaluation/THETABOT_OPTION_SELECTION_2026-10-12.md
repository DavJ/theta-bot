# ThetaBot: filtr očekávané opční výplaty — 12. 10. 2026

**Jde o modelový výzkum na již opakovaně použitých datech. Pozitivní odhad výplaty není ověřená tržní výhoda.**
Nákup používá historický obchod s příplatkem, nikoli ověřenou ask nabídku nebo hloubku trhu.
Průběžné opční ceny a propady jsou syntetické. Live způsobilost modelu: false.

## Výsledek a kontrola přínosu

Volba zmrazená podle let 2022–2024 v šesti scénářích: pro účet 10 000 **`call_empirical10`**.
Plný modelový výnos +168.61%, CAGR +23.14%, nejhorší modelový propad ze šesti scénářů -18.44%.
Filtr může odmítnout opci, jejíž odhadovaná čistá výplata nepřevyšuje plné nákupní náklady o stanovenou rezervu.
Hotovost se při odmítnutí nespotřebuje ani nepřerozdělí do druhého kontraktu.

| Účet 10 000 | Nominální výnos | Náklady + zpoždění | Volatilita předpovědi ×0,75 | ×1,25 | Nejhorší model DD |
|---|---:|---:|---:|---:|---:|
| `options_off` | +108.82% | +84.20% | +108.82% | +108.82% | -21.16% |
| `call_current1` | +110.80% | +83.07% | +110.80% | +110.80% | -19.18% |
| `call_current2` | +131.62% | +90.67% | +131.62% | +131.62% | -19.56% |
| `call_warmup` | +170.88% | +90.67% | +170.88% | +170.88% | -19.41% |
| `call_empirical10` | +168.61% | +98.87% | +132.41% | +170.88% | -18.44% |
| `direction_empirical10` | +172.75% | +100.77% | +127.81% | +177.04% | -18.57% |

**Samotná kontrola s čekáním na historii dává +170.88%, více než zvolený filtr.**
Proto nominální zvýšení proti původní call strategii nelze připsat pouze předpovědi. Filtr v tomto běhu
zlepšuje společný nákladový/časový stres a modelový propad vůči této kontrole; statisticky nezávislý důkaz nemáme.

Zvolený filtr má 19 nákupů a 7 ziskových opcí.
Tři největší výhry tvoří 92.96 % čistého opčního P&L.
Při jejich nezískání klesá výnos celého účtu na +101.42%,
oproti +108.82% spotu. Závislost na výhrách zůstává zásadní.
Směrový empirický filtr při vynechání svých tří největších výher dává +124.30%;
je však pouze další popisnou variantou a jeho odhad je citlivý na předpokládanou volatilitu.

U účtu 1 000 žádná inverse varianta nekoupila opci při povolených lotech a rozpočtu. Výběr zůstává spot;
ten sám při společném stresu překračuje 20% historický rozpočet propadu.

## Co přesně předpovídáme

[Zmrazený protokol](OPTION_FORECAST_RESEARCH_PLAN.md), SHA-256 `4d75ddc7257d02693184cab21df0c71a71019dad511a729c47ed87f86e726053`.

- Pouze dokončené denní závěry před UTC půlnocí dne nákupu; minimálně 252, nejvýše 730.
- Empirický model používá minulé překrývající se H-denní logaritmické výnosy se stejnými vahami.
- RV model používá volatilitu posledních 60 denních výnosů a lognormální rozdělení s nulovým driftem ceny.
- H je počet kalendářních dnů do známé expirace, zaokrouhlený nahoru. Skutečné plnění/vypořádání zůstává 4h.
- Odhadovaná výplata odečítá delivery a převodní poplatky. Předpověď předpokládá shodu indexu a spotu.
  Skutečný modelový cash flow dál používá oficiální delivery index a pozorovaný spotový převod.
- Výhoda 10/25 % se vztahuje k ceně prémie s náklady, nikoli k celému účtu ani k jistému výnosu.
- Rozpočet nákupů: nejvýše 2 % aktuálního financovaného účtu a současně 20 % opční hotovosti.
- Žádné opční ztráty se nedoplňují ze spotu. Opce se nevypisují a nevzniká dluh.
- Snížení/zvýšení volatility předpovědi mění volbu nákupů. Změna IV pouze pro držené značky nemění cash profit.

Development výběr maximalizuje nejhorší zisk v šesti scénářích při kladném zisku a DD <=20 % ve všech.
Volbu ukládáme před pozdějšími přepočty. Pozdější data jsme však už studovali; není to nový holdout.

## Všechny varianty: účet 1000

Zmrazená volba `options_off`; plný šestiscénářový rizikový screen: **False**.

| Varianta | Plný výnos 2022–2026 | CAGR | Samostatný 2025–2026 | Nejhorší ze 6 výnosů | Nejhorší model DD |
|---|---:|---:|---:|---:|---:|
| `options_off` | +110.26% | +16.95% | +35.45% | +84.16% | -21.30% |
| `reserved_cash` | +99.27% | +15.63% | +31.88% | +75.60% | -20.16% |
| `call_current1` | +99.27% | +15.63% | +31.88% | +75.60% | -20.16% |
| `call_current2` | +99.27% | +15.63% | +31.88% | +75.60% | -20.16% |
| `direction_current2` | +99.27% | +15.63% | +31.88% | +75.60% | -20.16% |
| `call_warmup` | +99.27% | +15.63% | +31.88% | +75.60% | -20.16% |
| `call_empirical10` | +99.27% | +15.63% | +31.88% | +75.60% | -20.16% |
| `call_empirical25` | +99.27% | +15.63% | +31.88% | +75.60% | -20.16% |
| `call_rv10` | +99.27% | +15.63% | +31.88% | +75.60% | -20.16% |
| `direction_warmup` | +99.27% | +15.63% | +31.88% | +75.60% | -20.16% |
| `direction_empirical10` | +99.27% | +15.63% | +31.88% | +75.60% | -20.16% |
| `direction_empirical25` | +99.27% | +15.63% | +31.88% | +75.60% | -20.16% |
| `direction_rv10` | +99.27% | +15.63% | +31.88% | +75.60% | -20.16% |

Roční účty začínají samostatně; jejich procenta se nesčítají. Počty nákupů/odmítnutí jsou za plnou historii.

| Varianta | Rok 2025 | Leden–září 2026 | Nákupů / výher | Odmítnutí / z toho krátká historie | Bez top 3 plnění |
|---|---:|---:|---|---|---:|
| `options_off` | +21.04% | +12.02% | 0 / 0 | 0 / 0 | — |
| `reserved_cash` | +18.82% | +11.01% | 0 / 0 | 0 / 0 | — |
| `call_current1` | +18.82% | +11.01% | 0 / 0 | 0 / 0 | +99.27% |
| `call_current2` | +18.82% | +11.01% | 0 / 0 | 0 / 0 | +99.27% |
| `direction_current2` | +18.82% | +11.01% | 0 / 0 | 0 / 0 | +99.27% |
| `call_warmup` | +18.82% | +11.01% | 0 / 0 | 4 / 4 | +99.27% |
| `call_empirical10` | +18.82% | +11.01% | 0 / 0 | 22 / 4 | +99.27% |
| `call_empirical25` | +18.82% | +11.01% | 0 / 0 | 25 / 4 | +99.27% |
| `call_rv10` | +18.82% | +11.01% | 0 / 0 | 39 / 4 | +99.27% |
| `direction_warmup` | +18.82% | +11.01% | 0 / 0 | 11 / 11 | +99.27% |
| `direction_empirical10` | +18.82% | +11.01% | 0 / 0 | 48 / 11 | +99.27% |
| `direction_empirical25` | +18.82% | +11.01% | 0 / 0 | 53 / 11 | +99.27% |
| `direction_rv10` | +18.82% | +11.01% | 0 / 0 | 70 / 11 | +99.27% |

### Samostatný nákladový a časový stres

| Varianta | Zdvojené sazby a vyšší prémie | +4h zpoždění | Náklady + zpoždění |
|---|---:|---:|---:|
| `options_off` | +93.80% | +99.54% | +84.16% |
| `reserved_cash` | +84.61% | +89.85% | +75.60% |
| `call_current1` | +84.61% | +89.85% | +75.60% |
| `call_current2` | +84.61% | +89.85% | +75.60% |
| `direction_current2` | +84.61% | +89.85% | +75.60% |
| `call_warmup` | +84.61% | +89.85% | +75.60% |
| `call_empirical10` | +84.61% | +89.85% | +75.60% |
| `call_empirical25` | +84.61% | +89.85% | +75.60% |
| `call_rv10` | +84.61% | +89.85% | +75.60% |
| `direction_warmup` | +84.61% | +89.85% | +75.60% |
| `direction_empirical10` | +84.61% | +89.85% | +75.60% |
| `direction_empirical25` | +84.61% | +89.85% | +75.60% |
| `direction_rv10` | +84.61% | +89.85% | +75.60% |

### Citlivost syntetických držených značek

| Varianta | Poloviční IV: DD | Dvojnásobná IV: DD |
|---|---:|---:|
| `call_current1` | -17.69% | -17.69% |
| `call_current2` | -17.69% | -17.69% |
| `direction_current2` | -17.69% | -17.69% |
| `call_warmup` | -17.69% | -17.69% |
| `call_empirical10` | -17.69% | -17.69% |
| `call_empirical25` | -17.69% | -17.69% |
| `call_rv10` | -17.69% | -17.69% |
| `direction_warmup` | -17.69% | -17.69% |
| `direction_empirical10` | -17.69% | -17.69% |
| `direction_empirical25` | -17.69% | -17.69% |
| `direction_rv10` | -17.69% | -17.69% |

Koncový zisk při těchto změnách IV zůstal stejný. Průběžný propad není ověřenou historickou mark cenou.

## Všechny varianty: účet 10000

Zmrazená volba `call_empirical10`; plný šestiscénářový rizikový screen: **True**.

| Varianta | Plný výnos 2022–2026 | CAGR | Samostatný 2025–2026 | Nejhorší ze 6 výnosů | Nejhorší model DD |
|---|---:|---:|---:|---:|---:|
| `options_off` | +108.82% | +16.78% | +35.31% | +84.20% | -21.16% |
| `reserved_cash` | +98.01% | +15.48% | +31.84% | +75.73% | -20.01% |
| `call_current1` | +110.80% | +17.01% | +36.30% | +83.07% | -19.18% |
| `call_current2` | +131.62% | +19.35% | +52.60% | +90.67% | -19.56% |
| `direction_current2` | +176.13% | +23.86% | +53.44% | +100.74% | -19.89% |
| `call_warmup` | +170.88% | +23.36% | +52.60% | +90.67% | -19.41% |
| `call_empirical10` | +168.61% | +23.14% | +52.43% | +98.87% | -18.44% |
| `call_empirical25` | +140.61% | +20.32% | +52.43% | +101.54% | -18.26% |
| `call_rv10` | +149.68% | +21.26% | +40.93% | +98.01% | -17.52% |
| `direction_warmup` | +175.63% | +23.81% | +53.44% | +101.83% | -19.34% |
| `direction_empirical10` | +172.75% | +23.54% | +53.86% | +100.77% | -18.57% |
| `direction_empirical25` | +134.40% | +19.65% | +53.86% | +95.14% | -18.93% |
| `direction_rv10` | +136.40% | +19.87% | +35.62% | +91.67% | -18.68% |

Roční účty začínají samostatně; jejich procenta se nesčítají. Počty nákupů/odmítnutí jsou za plnou historii.

| Varianta | Rok 2025 | Leden–září 2026 | Nákupů / výher | Odmítnutí / z toho krátká historie | Bez top 3 plnění |
|---|---:|---:|---|---|---:|
| `options_off` | +20.91% | +11.95% | 0 / 0 | 0 / 0 | — |
| `reserved_cash` | +18.84% | +10.75% | 0 / 0 | 0 / 0 | — |
| `call_current1` | +18.84% | +10.75% | 5 / 2 | 0 / 0 | +96.56% |
| `call_current2` | +18.84% | +15.21% | 17 / 6 | 0 / 0 | +93.53% |
| `direction_current2` | +17.25% | +29.47% | 38 / 15 | 0 / 0 | +98.73% |
| `call_warmup` | +18.84% | +15.21% | 24 / 10 | 4 / 4 | +97.31% |
| `call_empirical10` | +18.84% | +15.21% | 19 / 7 | 22 / 4 | +101.42% |
| `call_empirical25` | +18.84% | +15.21% | 16 / 5 | 25 / 4 | +99.14% |
| `call_rv10` | +18.84% | +15.21% | 8 / 6 | 39 / 4 | +96.45% |
| `direction_warmup` | +17.25% | +29.47% | 39 / 15 | 11 / 11 | +98.23% |
| `direction_empirical10` | +18.84% | +29.47% | 25 / 10 | 48 / 11 | +124.30% |
| `direction_empirical25` | +18.84% | +29.47% | 21 / 7 | 53 / 11 | +100.54% |
| `direction_rv10` | +18.15% | +15.21% | 15 / 8 | 70 / 11 | +98.65% |

### Samostatný nákladový a časový stres

| Varianta | Zdvojené sazby a vyšší prémie | +4h zpoždění | Náklady + zpoždění |
|---|---:|---:|---:|
| `options_off` | +92.24% | +100.09% | +84.20% |
| `reserved_cash` | +83.06% | +90.05% | +75.73% |
| `call_current1` | +96.70% | +114.14% | +83.07% |
| `call_current2` | +110.37% | +102.94% | +90.67% |
| `direction_current2` | +131.32% | +138.76% | +100.74% |
| `call_warmup` | +110.68% | +104.16% | +90.67% |
| `call_empirical10` | +113.25% | +137.31% | +98.87% |
| `call_empirical25` | +125.91% | +138.80% | +107.24% |
| `call_rv10` | +125.83% | +129.83% | +101.85% |
| `direction_warmup` | +131.32% | +124.81% | +101.83% |
| `direction_empirical10` | +115.53% | +131.02% | +100.77% |
| `direction_empirical25` | +120.69% | +124.33% | +95.14% |
| `direction_rv10` | +113.88% | +105.42% | +91.67% |

### Citlivost syntetických držených značek

| Varianta | Poloviční IV: DD | Dvojnásobná IV: DD |
|---|---:|---:|
| `call_current1` | -16.81% | -16.81% |
| `call_current2` | -18.54% | -17.64% |
| `direction_current2` | -18.10% | -17.38% |
| `call_warmup` | -17.15% | -17.59% |
| `call_empirical10` | -16.46% | -15.87% |
| `call_empirical25` | -16.43% | -16.43% |
| `call_rv10` | -16.33% | -15.87% |
| `direction_warmup` | -18.14% | -17.96% |
| `direction_empirical10` | -17.10% | -16.74% |
| `direction_empirical25` | -17.48% | -17.48% |
| `direction_rv10` | -16.94% | -15.97% |

Koncový zisk při těchto změnách IV zůstal stejný. Průběžný propad není ověřenou historickou mark cenou.

## Náklady a krátký horizont: účet 10 000

| Varianta | Opční poplatky | Převodní poplatky | Převodní skluz | Příplatek nad cenu obchodu |
|---|---:|---:|---:|---:|
| `call_current1` | 5.07 | 1.84 | 1.11 | 15.94 |
| `call_current2` | 25.26 | 7.47 | 4.49 | 112.01 |
| `direction_current2` | 57.36 | 15.80 | 9.48 | 223.77 |
| `call_warmup` | 48.53 | 14.02 | 8.41 | 189.08 |
| `call_empirical10` | 44.32 | 12.82 | 7.69 | 163.35 |
| `call_empirical25` | 34.72 | 9.03 | 5.42 | 134.29 |
| `call_rv10` | 22.21 | 7.91 | 4.75 | 77.89 |
| `direction_warmup` | 57.92 | 15.85 | 9.51 | 225.77 |
| `direction_empirical10` | 44.40 | 12.56 | 7.54 | 148.01 |
| `direction_empirical25` | 33.49 | 7.31 | 4.39 | 107.70 |
| `direction_rv10` | 26.61 | 7.03 | 4.22 | 91.33 |

USD-equivalent za nominální běh; spotové náklady a všechny ostatní scénáře jsou v JSON.

| Varianta | Nejlepší 7 dní | Nejlepší 30 dní | Nejlepší 90 dní | Zdvojnásobení 7 / 30 / 90 |
|---|---:|---:|---:|---|
| `options_off` | +11.46% | +26.97% | +29.52% | 0 / 0 / 0 |
| `call_warmup` | +29.84% | +35.29% | +43.63% | 0 / 0 / 0 |
| `call_empirical10` | +30.18% | +38.53% | +47.24% | 0 / 0 / 0 |
| `direction_empirical10` | +29.56% | +31.45% | +40.85% | 0 / 0 / 0 |

Okna se překrývají a používají syntetické opční značky; nejde o realizované ani garantované budoucí výnosy.

## Ověření a reprodukce

Účetně ověřeno 846 cest; NAV reziduum 2.55e-11. 240 kontrol zachovalo archivované ekonomické hodnoty.
Nezávislý audit: 227 uzavřených nákupů, 944 měsíčních rozpočtů a 1064 rozhodnutí.
Reziduum cash záznamů 9.09e-13; rozhodnutí/nákupních debetů 4.55e-13.
Audit kontroluje cut-off, dokončenost dne, známou expiraci, minimální historii, fixní práh výhody a soulad s nakoupeným množstvím.
Surové zdroje: 39 souborů, 550 obchodních oken; původní pozorování hash `2bce888f28861aba42e94a044279e3ea112b1fd33e7c8219fa3769a5125bf9a6`.

```bash
python -m scripts.download_spot_flow
python -m scripts.download_long_options
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m scripts.research_option_selection --options-enabled
python -m scripts.write_option_selection_report
python -m pytest -q tests
```

[THETABOT_OPTION_SELECTION_2026-10-12.json.gz](THETABOT_OPTION_SELECTION_2026-10-12.json.gz) — úplný JSON komprimovaný gzip: všechny varianty, scénáře, volby a limity.
[THETABOT_OPTION_SELECTION_2026-10-12_CLOSED_OPTIONS.csv](THETABOT_OPTION_SELECTION_2026-10-12_CLOSED_OPTIONS.csv)
[THETABOT_OPTION_SELECTION_2026-10-12_PURCHASE_BUDGETS.csv](THETABOT_OPTION_SELECTION_2026-10-12_PURCHASE_BUDGETS.csv)
[THETABOT_OPTION_SELECTION_2026-10-12_ENTRY_SELECTION.csv](THETABOT_OPTION_SELECTION_2026-10-12_ENTRY_SELECTION.csv)
Evidence SHA-256 `3e7db587ec5386a4f9739791a6ec5bc4e53f057a1fded8c1abfea23c7e1fd65c`.

## Omezení

- Trade prices are not historical executable bid/ask quotes or market depth.
- Constant-entry-IV Black-Scholes held values are synthetic, including model drawdowns.
- Uniform current fee scenarios and conservative lot floors are not dated rule histories.
- Binance spot converts inverse premiums/payoffs; index/venue basis remains.
- USD/USDT/USDC parity, stablecoin and exchange failure risk are not simulated.
- These spot dates and momentum signals were studied before; not an unseen holdout.
- A fixed 10% initial option sleeve is not replenished; wins can change its later weight.
- A 20% historical model drawdown is not a future loss guarantee.
- Entry debit limits do not guarantee a cap on held option market values or future drawdown.
- Missed top-three fills are selected ex post as a fragility diagnostic, not an entry predictor.
- Entry debit limits do not guarantee a cap on held option market values or future drawdown.
- Missed top-three fills are selected ex post as a fragility diagnostic, not an entry predictor.
- Forecast models assume conditional delivery-index/spot parity and uncalibrated return distributions.
- Overlapping historical return windows are not independent samples.
- Forecast volatility scaling changes entry decisions; it is separate from synthetic held-mark IV sensitivity.

Primární specifikace: [původní opční protokol](LONG_OPTIONS_RESEARCH_PLAN.md).
