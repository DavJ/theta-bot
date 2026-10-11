# ThetaBot: omezení opčních nákupů — 12. 10. 2026

**Jde o další modelový výzkum na již použitých datech. Nákupní ceny nejsou ověřené historické ask nabídky.**
Průběžné opční ceny i propady jsou syntetické. Výsledek není způsobilý pro živé obchodování.

## Co se zlepšilo a co zůstává slabé

Do výzkumného enginu přibylo omezení všech měsíčních nákupních debetů podle skutečně financovaného účtu.
Platí pro prémii, obchodní poplatky, převodní poplatky i skluz dohromady. Výhry v opční části samy nemohou
zvýšit příští nákup nad nastavené procento celého účtu. Spot opční ztráty nedoplňuje.

U účtu 10 000 směrová varianta s limitem 2 % aktuálního účtu dává modelově +176.13%
oproti původním +314.31%. Nejvyšší vykázaný měsíční nákupní debet v modelu klesl
z 5.98 % na 1.89 % účtu.
Vyšší náklady a zpoždění nyní stresujeme i současně. Menší nominální profit je cena za omezené sázky.

| Účet 10 000: varianta | Základní výnos | Vyšší náklady | +4h zpoždění | Náklady a zpoždění současně | Nejhorší model DD ze 4 scénářů |
|---|---:|---:|---:|---:|---:|
| `options_off` | +108.82% | +92.24% | +100.09% | +84.20% | -21.16% |
| `legacy_direction_otm` | +314.31% | +236.23% | +132.15% | +85.09% | -20.14% |
| `direction_current2` | +176.13% | +131.32% | +138.76% | +100.74% | -19.89% |
| `direction_initial2` | +129.26% | +98.22% | +123.00% | +102.16% | -18.47% |
| `call_current1` | +110.80% | +96.70% | +114.14% | +83.07% | -19.18% |

**Volba zmrazená podle vývoje 2022–2024 je `call_current1`**, nikoli dodatečně vybraná nejvýnosnější
varianta celého období. Směrová varianta s 2% limitem je popisný kandidát k dalšímu ověření.
Tato časová hranice není nezávislý holdout: výsledky pozdějších let jsme již znali z předchozí práce.

Bez získání tří nejziskovějších nákupů klesá směrová 2% varianta na +98.73%,
zatímco samotný spot dává +108.82%. Koncentrace klesla z
80.04 % na 65.23 %
čistého opčního P&L, ale závislost na několika výhrách se nevyřešila. Vynechání je ex-post diagnostika
selhání plnění; nepredikuje, které budoucí obchody vynechat.

Pro účet 1 000 žádná z těchto inverse variant nekoupila opci při povolených minimálních lotech.
Rezervovaný kapitál snižuje spotový výnos. I původní spot překračuje 20 % při společném stresu.

## Přesná pravidla

[Zmrazený protokol](OPTION_BUDGET_RESEARCH_PLAN.md), SHA-256 `691734638d4cf81ca5b8ba343bf15363f69749483bf35d0b18a2d4d05927e01f`.

- Dva OTM signály: call-only při kladném 30denním momentu; call/put podle znaménka momenta.
- Pro každý signál tři limity: 2 % aktuálního účtu, 1 % aktuálního účtu, 2 % počátečního kapitálu.
- Rozpočet je vždy také omezen 20 % aktuální opční hotovosti. Použije se menší limit.
- Počáteční financování: 90 % spot / 10 % opční hotovost. Bez převodů ze spotu a bez dluhu.
- Aktuální hodnota pro rozpočet: předchozí vlastní spotová pozice a hotovost při nynějším open,
  plus opční hotovost. Aktuální close ani pozdější pozice velikost nákupu neovlivňují.
- Nákupní limit neomezuje tržní cenu již držené opce a negarantuje budoucí maximální propad.
- Kalendář, kontrakty, ocenění při nákupu/expiraci, lotech a sazby: původní opční protokol.
- Volba na development: kladný zisk a DD <=20 % ve všech 4 scénářích; maximalizovat nejhorší zisk.
- Nominální +5% příplatek k obchodu; nákladový stres +20% příplatek a zdvojené sazby; zpoždění 4h.
- Chybějící a nedostupné kontrakty se nepřesouvají do jiných kontraktů nebo umělých zlomků.

## Všechny varianty: účet 1000

Zmrazená volba: `options_off`. Plný čtyřscénářový rizikový screen: **False**.
Výnos je za celý účet, leden 2022–září 2026. Pozdější účet začíná samostatně v lednu 2025.

| Varianta | Plný zisk | CAGR | Zisk 2025–2026 | Nejhorší ze 4 zisků | Nejhorší model DD |
|---|---:|---:|---:|---:|---:|
| `options_off` | +110.26% | +16.95% | +35.45% | +84.16% | -21.30% |
| `reserved_cash` | +99.27% | +15.63% | +31.88% | +75.60% | -20.16% |
| `legacy_call_otm` | +99.27% | +15.63% | +31.88% | +75.60% | -20.16% |
| `legacy_direction_otm` | +99.27% | +15.63% | +31.88% | +75.60% | -20.16% |
| `call_current2` | +99.27% | +15.63% | +31.88% | +75.60% | -20.16% |
| `call_current1` | +99.27% | +15.63% | +31.88% | +75.60% | -20.16% |
| `call_initial2` | +99.27% | +15.63% | +31.88% | +75.60% | -20.16% |
| `direction_current2` | +99.27% | +15.63% | +31.88% | +75.60% | -20.16% |
| `direction_current1` | +99.27% | +15.63% | +31.88% | +75.60% | -20.16% |
| `direction_initial2` | +99.27% | +15.63% | +31.88% | +75.60% | -20.16% |

| Varianta | Samostatný 2025 | Leden–září 2026 | Nákupů / výher za celou historii | Malý lot / chybějící obchod |
|---|---:|---:|---|---|
| `options_off` | +21.04% | +12.02% | 0 / 0 | 0 / 0 |
| `reserved_cash` | +18.82% | +11.01% | 0 / 0 | 0 / 0 |
| `legacy_call_otm` | +18.82% | +11.01% | 0 / 0 | 47 / 12 |
| `legacy_direction_otm` | +18.82% | +11.01% | 0 / 0 | 86 / 26 |
| `call_current2` | +18.82% | +11.01% | 0 / 0 | 47 / 12 |
| `call_current1` | +18.82% | +11.01% | 0 / 0 | 47 / 12 |
| `call_initial2` | +18.82% | +11.01% | 0 / 0 | 47 / 12 |
| `direction_current2` | +18.82% | +11.01% | 0 / 0 | 86 / 26 |
| `direction_current1` | +18.82% | +11.01% | 0 / 0 | 86 / 26 |
| `direction_initial2` | +18.82% | +11.01% | 0 / 0 | 86 / 26 |

### Bez nejlepších plnění a citlivost oceňování

| Varianta | Zisk při vynechání top 3 výher | Poloviční IV: DD | Dvojnásobná IV: DD |
|---|---:|---:|---:|
| `legacy_call_otm` | +99.27% | -17.69% | -17.69% |
| `legacy_direction_otm` | +99.27% | -17.69% | -17.69% |
| `call_current2` | +99.27% | -17.69% | -17.69% |
| `call_current1` | +99.27% | -17.69% | -17.69% |
| `call_initial2` | +99.27% | -17.69% | -17.69% |
| `direction_current2` | +99.27% | -17.69% | -17.69% |
| `direction_current1` | +99.27% | -17.69% | -17.69% |
| `direction_initial2` | +99.27% | -17.69% | -17.69% |

Při změně IV zůstaly výplaty při expiraci i koncový zisk stejné. Propady zůstávají modelové.

## Všechny varianty: účet 10000

Zmrazená volba: `call_current1`. Plný čtyřscénářový rizikový screen: **True**.
Výnos je za celý účet, leden 2022–září 2026. Pozdější účet začíná samostatně v lednu 2025.

| Varianta | Plný zisk | CAGR | Zisk 2025–2026 | Nejhorší ze 4 zisků | Nejhorší model DD |
|---|---:|---:|---:|---:|---:|
| `options_off` | +108.82% | +16.78% | +35.31% | +84.20% | -21.16% |
| `reserved_cash` | +98.01% | +15.48% | +31.84% | +75.73% | -20.01% |
| `legacy_call_otm` | +129.75% | +19.15% | +52.60% | +88.67% | -19.56% |
| `legacy_direction_otm` | +314.31% | +34.91% | +53.44% | +85.09% | -20.14% |
| `call_current2` | +131.62% | +19.35% | +52.60% | +90.67% | -19.56% |
| `call_current1` | +110.80% | +17.01% | +36.30% | +83.07% | -19.18% |
| `call_initial2` | +123.43% | +18.45% | +52.60% | +91.13% | -19.55% |
| `direction_current2` | +176.13% | +23.86% | +53.44% | +100.74% | -19.89% |
| `direction_current1` | +109.46% | +16.85% | +36.30% | +81.32% | -19.36% |
| `direction_initial2` | +129.26% | +19.10% | +53.44% | +98.22% | -18.47% |

| Varianta | Samostatný 2025 | Leden–září 2026 | Nákupů / výher za celou historii | Malý lot / chybějící obchod |
|---|---:|---:|---|---|
| `options_off` | +20.91% | +11.95% | 0 / 0 | 0 / 0 |
| `reserved_cash` | +18.84% | +10.75% | 0 / 0 | 0 / 0 |
| `legacy_call_otm` | +18.84% | +31.51% | 18 / 6 | 29 / 12 |
| `legacy_direction_otm` | +17.25% | +32.93% | 50 / 19 | 36 / 26 |
| `call_current2` | +18.84% | +15.21% | 17 / 6 | 30 / 12 |
| `call_current1` | +18.84% | +10.75% | 5 / 2 | 42 / 12 |
| `call_initial2` | +18.84% | +31.51% | 11 / 4 | 36 / 12 |
| `direction_current2` | +17.25% | +29.47% | 38 / 15 | 48 / 26 |
| `direction_current1` | +18.84% | +10.75% | 13 / 5 | 73 / 26 |
| `direction_initial2` | +17.25% | +34.10% | 24 / 9 | 62 / 26 |

### Bez nejlepších plnění a citlivost oceňování

| Varianta | Zisk při vynechání top 3 výher | Poloviční IV: DD | Dvojnásobná IV: DD |
|---|---:|---:|---:|
| `legacy_call_otm` | +93.53% | -18.54% | -17.64% |
| `legacy_direction_otm` | +143.52% | -19.65% | -18.34% |
| `call_current2` | +93.53% | -18.54% | -17.64% |
| `call_current1` | +96.56% | -16.81% | -16.81% |
| `call_initial2` | +93.53% | -17.23% | -17.12% |
| `direction_current2` | +98.73% | -18.10% | -17.38% |
| `direction_current1` | +97.52% | -17.14% | -17.14% |
| `direction_initial2` | +98.73% | -17.35% | -16.96% |

Při změně IV zůstaly výplaty při expiraci i koncový zisk stejné. Propady zůstávají modelové.

## Náklady a krátký horizont: účet 10 000

| Varianta | Poplatky opcí | Poplatky převodu | Skluz převodu | Příplatek nad původní prémii |
|---|---:|---:|---:|---:|
| `legacy_call_otm` | 27.62 | 7.66 | 4.60 | 123.79 |
| `legacy_direction_otm` | 150.80 | 43.64 | 26.19 | 603.74 |
| `call_current2` | 25.26 | 7.47 | 4.49 | 112.01 |
| `call_current1` | 5.07 | 1.84 | 1.11 | 15.94 |
| `call_initial2` | 13.12 | 4.33 | 2.60 | 48.48 |
| `direction_current2` | 57.36 | 15.80 | 9.48 | 223.77 |
| `direction_current1` | 10.88 | 2.35 | 1.41 | 34.60 |
| `direction_initial2` | 24.85 | 6.42 | 3.85 | 92.79 |

Částky v USD-equivalentu za plný nominální běh; spotové náklady jsou zvlášť v JSON.

| Varianta | Nejlepší 7 dní | Nejlepší 30 dní | Nejlepší 90 dní | Zdvojnásobení 7 / 30 / 90 |
|---|---:|---:|---:|---|
| `options_off` | +11.46% | +26.97% | +29.52% | 0 / 0 / 0 |
| `legacy_direction_otm` | +73.34% | +77.11% | +102.29% | 0 / 0 / 1 |
| `direction_current2` | +29.08% | +30.94% | +40.14% | 0 / 0 / 0 |
| `call_current1` | +12.87% | +29.39% | +35.62% | 0 / 0 / 0 |

Denní překrývající se okna zahrnují syntetické ceny držených opcí. Nejsou realizovanou výplatou ani odhadem budoucí četnosti.

## Ověření a reprodukce

Účetně ověřeno 448 cest, maximální NAV reziduum 2.55e-11.
Celkem 96 porovnání kontrol zachovalo všechna archivovaná ekonomická pole.
Samostatně ověřeno 176 uzavřených nákupů a 528 měsíčních rozpočtů; reziduum 3.64e-12.
Ověřeno 39 surových zdrojových souborů a 550 historických obchodních oken.
Původní pozorování SHA-256 `2bce888f28861aba42e94a044279e3ea112b1fd33e7c8219fa3769a5125bf9a6`.

```bash
python -m scripts.download_spot_flow
python -m scripts.download_long_options
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m scripts.research_option_budgets --options-enabled
python -m scripts.write_option_budgets_report
python -m pytest -q tests
```

[THETABOT_OPTION_BUDGETS_2026-10-12.json](THETABOT_OPTION_BUDGETS_2026-10-12.json) — všechny varianty, období, scénáře, zmrazená volba a vynechané vstupy.
[THETABOT_OPTION_BUDGETS_2026-10-12_CLOSED_OPTIONS.csv](THETABOT_OPTION_BUDGETS_2026-10-12_CLOSED_OPTIONS.csv) — ceny, náklady, výplaty, ID/čas obchodu.
[THETABOT_OPTION_BUDGETS_2026-10-12_PURCHASE_BUDGETS.csv](THETABOT_OPTION_BUDGETS_2026-10-12_PURCHASE_BUDGETS.csv) — hotovost, základ limitu, povolený a skutečný měsíční debet.
JSON SHA-256 `af81b4949fc06cc1a1e3c66e0b840940c09d9dbe00ea7b58b12d0300ab6157f7`.

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

Primární specifikace: [původní protokol](LONG_OPTIONS_RESEARCH_PLAN.md).
