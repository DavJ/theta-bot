# ThetaBot: model ziskovosti koupených opcí — 10. 10. 2026

**Výnosy jsou modelové. Vstupní ceny vycházejí z historických obchodů, nikoli z ověřených nabídek k nákupu.**
Hodnota držených opcí a průběžné propady používají syntetický oceňovací model.
Výsledek není ověřeným proveditelným opčním backtestem. Live způsobilost: **false**.

## Výsledek

Pro účet 1 000 jednotek opce nepřekonaly původní spot. Inverse kontrakty nebyly při stanoveném rozpočtu dostupné.
Pro účet 10 000 byla před pozdějšími výpočty vybrána **`inverse_direction_otm`**:
modelový čistý výnos celého účtu +314.31%, oproti +108.82% bez opcí.
Pozdější samostatně financovaný účet 2025–2026 dává podstatně menší výnos; nesmí se zaměňovat s plnou historií.

| Vybraná opční varianta, účet 10 000 | Čistý výnos 2022–2026 | Modelový propad |
|---|---:|---:|
| Základní model | +314.31% | -18.38% |
| Vyšší prémie a dvojnásobné náklady | +236.23% | -20.14% |
| Zpoždění nákupu o 4 hodiny | +132.15% | -19.67% |

**Kandidát neprošel všemi scénáři 20% rozpočtu modelového propadu.**
Tři nejziskovější opce tvoří 80.04 % čistého opčního P&L. Výsledek je koncentrovaný do malého počtu výplat a citlivý na dostupnost vstupní ceny.
Opční část na konci tvoří 54.62 % účtu.
**Počáteční 10% oddělení kapitálu není trvalý 10% strop expozice. Počáteční 2% nákupní limit také není trvalý 2% limit účtu.**
Zisky zůstávají v opční části, ale ze spotu se do ní další peníze nepřevádějí.

### Krátký horizont bez otevřených opcí na koncích okna

Doplňková kontrola vybrané varianty a účtu 10 000: oba koncové denní body musí mít všechny opce vypořádané.
Spot zůstává oceněn tržně. Tato okna nemají syntetickou opční cenu v počáteční ani konečné hodnotě.

| Délka | Dostupných oken | Nejlepší výnos | Zdvojnásobení |
|---|---:|---:|---:|
| 7 dní | 678 | +8.44% | 0 |
| 30 dní | 520 | +71.26% | 0 |
| 90 dní | 535 | +94.04% | 0 |

Jde o dodatečnou popisnou diagnostiku, nikoli nové pravidlo výběru nebo odhad budoucích výnosů.

## Co přesně porovnáváme

- Jeden účet od ledna 2022 do září 2026; výnosy zahrnují složené úročení a modelové náklady.
- Vypnuto: původní capped spot strategie s celým kapitálem. Rezerva: 90 % spot / 10 % hotovost.
- Zapnuto: 90 % spot / 10 % oddělená opční část, bez doplňování ztrát ze spotu.
- Každý měsíc první pondělí: nákup do posledního pátečního vypořádání měsíce.
- Prémie a všechny náklady nákupu celkem nejvýše 20 % aktuální opční hotovosti.
  Zpočátku tedy nejvýše 2 % celého účtu. Rozpočet může po opčních ziscích růst.
- BTC/ETH: call při kladném 30denním momentu, nebo call/put podle jeho znaménka;
  strike nejblíže poslednímu dokončenému dennímu závěru (ATM), nebo s cílem o 5 % mimo peníze (OTM). Opce pouze kupujeme.
- Žádné půjčky, vypisování opcí, samostatné futures, martingale ani živé objednávky.
- Inverse ceny/výplaty v BTC/ETH se převádějí za vlastní hotovost; linear jsou v USDC.
- USDC BTC/ETH produkty existují až od srpna 2025. Před dostupností zůstává jejich část v hotovosti.
- Minimální velikost kontraktu respektujeme; nepřístupný nákup se přeskočí.

Předem zmrazený [protokol](LONG_OPTIONS_RESEARCH_PLAN.md), SHA-256 `3c40130d2a1f30011d2e9c632a04162c475cf4311ccfb3785cc95dbad4cfe3fd`.
Studovaná spotová období už byla použita; nejde o nový nezávislý holdout.

## Model cen a nákladů

Poslední skutečný obchod ve vybraném kontraktu musí předcházet nákupu a být maximálně 4 hodiny starý.
Model zaplatí jeho prémii zvýšenou o 5 %, zaokrouhlenou nahoru na tick. Stres používá +20 %.
Chybějící cenu nenahrazujeme umělou prémií a nevybíráme dodatečně jiný kontrakt.
Nemáme historické bid/ask nabídky, dostupnou hloubku ani zaručené plnění.
Zaokrouhlení na tick může zvýšit skutečný modelový příplatek nad 5 %.

Nákupní poplatek je scénář 0,03 % podkladového objemu, nejvýše 12,5 % prémie.
Vypořádání ziskové opce: 0,015 % podkladového objemu, nejvýše 12,5 % výplaty.
Inverse převody: 0,1 % poplatek + 5 bps skluz + polovina 2 bps spreadu.
Linear převody: 0,1 % poplatek a předpoklad USD/USDT/USDC parity. Náklady se v účetnictví neodečítají dvakrát.
Stres zdvojnásobuje sazby; 12,5% strop zůstává. Jde o jednotný scénář podle současných pravidel,
nikoli rekonstrukci historických poplatků a jejich změn.

Výplata při expiraci používá oficiální Deribit delivery index. Průběžný Black-Scholes model má
nulovou sazbu a vstupní IV neměnnou do expirace. Spot OHLC slouží i pro modelové průběžné propady.
Všechny opce do konce období expirují: terminální výnos nezávisí na těchto syntetických značkách.

## Počáteční účet 1 000 jednotek

Volba zmrazená pouze podle vývoje 2022–2024: **`options_off`**.
Procenta se vztahují k celému účtu. DD je modelový 4h bound; nevydáváme ho za skutečný maximální propad.

| Varianta | Celý účet 2022–2026 | CAGR | 2025–2026 | Opční P&L | Model DD | Nákladový stres | Zpoždění +4h |
|---|---:|---:|---:|---:|---:|---:|---:|
| `options_off` | +110.26% | +16.95% | +35.45% | +0.00 | -18.62% | +93.80% | +99.54% |
| `reserved_cash` | +99.27% | +15.63% | +31.88% | +0.00 | -17.69% | +84.61% | +89.85% |
| `inverse_call_atm` | +99.27% | +15.63% | +31.88% | +0.00 | -17.69% | +84.61% | +89.85% |
| `inverse_call_otm` | +99.27% | +15.63% | +31.88% | +0.00 | -17.69% | +84.61% | +89.85% |
| `inverse_direction_atm` | +99.27% | +15.63% | +31.88% | +0.00 | -17.69% | +84.61% | +89.85% |
| `inverse_direction_otm` | +99.27% | +15.63% | +31.88% | +0.00 | -17.69% | +84.61% | +89.85% |
| `linear_call_atm` | +104.20% | +16.23% | +36.81% | +49.31 | -17.69% | +84.61% | +89.85% |
| `linear_call_otm` | +99.05% | +15.60% | +31.67% | -2.13 | -17.69% | +84.12% | +89.85% |
| `linear_direction_atm` | +104.20% | +16.23% | +36.81% | +49.31 | -17.69% | +84.61% | +89.85% |
| `linear_direction_otm` | +99.05% | +15.60% | +31.67% | -2.13 | -17.69% | +84.12% | +89.85% |

Samostatně začínající roční účty; nesčítat roční procenta. Počty nákupů, výher a přeskočení jsou za celou historii 2022–2026.

| Varianta | Rok 2025 | Leden–září 2026 | Opčních nákupů | Ziskových | Přeskočeno: malý rozpočet / bez obchodu / bez kontraktu |
|---|---:|---:|---:|---:|---|
| `options_off` | +21.04% | +12.02% | 0 | 0/0 | 0 / 0 / 0 |
| `reserved_cash` | +18.82% | +11.01% | 0 | 0/0 | 0 / 0 / 0 |
| `inverse_call_atm` | +18.82% | +11.01% | 0 | 0/0 | 48 / 11 / 0 |
| `inverse_call_otm` | +18.82% | +11.01% | 0 | 0/0 | 47 / 12 / 0 |
| `inverse_direction_atm` | +18.82% | +11.01% | 0 | 0/0 | 94 / 18 / 0 |
| `inverse_direction_otm` | +18.82% | +11.01% | 0 | 0/0 | 86 / 26 / 0 |
| `linear_call_atm` | +18.82% | +15.94% | 2 | 1/2 | 3 / 10 / 44 |
| `linear_call_otm` | +18.82% | +10.80% | 2 | 1/2 | 1 / 12 / 44 |
| `linear_direction_atm` | +18.82% | +15.94% | 2 | 1/2 | 3 / 21 / 86 |
| `linear_direction_otm` | +18.82% | +10.80% | 2 | 1/2 | 1 / 23 / 86 |

### Výnosy za kratší období

| Varianta | Nejlepší 7 dní | Nejlepší 30 dní | Nejlepší 90 dní | Zdvojnásobení 7 / 30 / 90 dní |
|---|---:|---:|---:|---|
| `options_off` | +11.58% | +27.19% | +29.74% | 0 / 0 / 0 |
| `reserved_cash` | +10.91% | +25.25% | +27.46% | 0 / 0 / 0 |
| `inverse_call_atm` | +10.91% | +25.25% | +27.46% | 0 / 0 / 0 |
| `inverse_call_otm` | +10.91% | +25.25% | +27.46% | 0 / 0 / 0 |
| `inverse_direction_atm` | +10.91% | +25.25% | +27.46% | 0 / 0 / 0 |
| `inverse_direction_otm` | +10.91% | +25.25% | +27.46% | 0 / 0 / 0 |
| `linear_call_atm` | +13.98% | +25.25% | +27.46% | 0 / 0 / 0 |
| `linear_call_otm` | +10.93% | +25.25% | +27.46% | 0 / 0 / 0 |
| `linear_direction_atm` | +13.98% | +25.25% | +27.46% | 0 / 0 / 0 |
| `linear_direction_otm` | +10.93% | +25.25% | +27.46% | 0 / 0 / 0 |

Okna se překrývají. Zisky a zdvojnásobení uvnitř doby držení závisí na modelových cenách opcí;
nejde o realizované výnosy ani budoucí pravděpodobnosti.

### Citlivost modelového propadu na IV

| Varianta | Poloviční IV | Vstupní IV | Dvojnásobná IV |
|---|---:|---:|---:|
| `inverse_call_atm` | -17.69% | -17.69% | -17.69% |
| `inverse_call_otm` | -17.69% | -17.69% | -17.69% |
| `inverse_direction_atm` | -17.69% | -17.69% | -17.69% |
| `inverse_direction_otm` | -17.69% | -17.69% | -17.69% |
| `linear_call_atm` | -17.69% | -17.69% | -17.69% |
| `linear_call_otm` | -17.80% | -17.69% | -17.69% |
| `linear_direction_atm` | -17.69% | -17.69% | -17.69% |
| `linear_direction_otm` | -17.80% | -17.69% | -17.69% |

Změna oceňovací IV nemění terminální opční cash flow ani zisk účtu; mění vykazovaný průběžný propad.

## Počáteční účet 10 000 jednotek

Volba zmrazená pouze podle vývoje 2022–2024: **`inverse_direction_otm`**.
Procenta se vztahují k celému účtu. DD je modelový 4h bound; nevydáváme ho za skutečný maximální propad.

| Varianta | Celý účet 2022–2026 | CAGR | 2025–2026 | Opční P&L | Model DD | Nákladový stres | Zpoždění +4h |
|---|---:|---:|---:|---:|---:|---:|---:|
| `options_off` | +108.82% | +16.78% | +35.31% | +0.00 | -18.44% | +92.24% | +100.09% |
| `reserved_cash` | +98.01% | +15.48% | +31.84% | +0.00 | -17.52% | +83.06% | +90.05% |
| `inverse_call_atm` | +107.52% | +16.62% | +31.84% | +951.54 | -16.71% | +87.81% | +165.56% |
| `inverse_call_otm` | +129.75% | +19.15% | +52.60% | +3174.14 | -18.28% | +107.53% | +102.46% |
| `inverse_direction_atm` | +96.03% | +15.23% | +31.84% | -198.18 | -17.69% | +81.91% | +107.22% |
| `inverse_direction_otm` | +314.31% | +34.91% | +53.44% | +21630.15 | -18.38% | +236.23% | +132.15% |
| `linear_call_atm` | +101.28% | +15.88% | +35.11% | +327.00 | -17.52% | +85.51% | +89.63% |
| `linear_call_otm` | +96.74% | +15.32% | +30.57% | -127.11 | -17.52% | +81.59% | +88.19% |
| `linear_direction_atm` | +101.28% | +15.88% | +35.11% | +327.00 | -17.52% | +85.51% | +91.90% |
| `linear_direction_otm` | +96.74% | +15.32% | +30.57% | -127.11 | -17.52% | +81.59% | +87.86% |

Samostatně začínající roční účty; nesčítat roční procenta. Počty nákupů, výher a přeskočení jsou za celou historii 2022–2026.

| Varianta | Rok 2025 | Leden–září 2026 | Opčních nákupů | Ziskových | Přeskočeno: malý rozpočet / bez obchodu / bez kontraktu |
|---|---:|---:|---:|---:|---|
| `options_off` | +20.91% | +11.95% | 0 | 0/0 | 0 / 0 / 0 |
| `reserved_cash` | +18.84% | +10.75% | 0 | 0/0 | 0 / 0 / 0 |
| `inverse_call_atm` | +18.84% | +10.75% | 6 | 2/6 | 42 / 11 / 0 |
| `inverse_call_otm` | +18.84% | +31.51% | 18 | 6/18 | 29 / 12 / 0 |
| `inverse_direction_atm` | +18.84% | +10.75% | 7 | 1/7 | 87 / 18 / 0 |
| `inverse_direction_otm` | +17.25% | +32.93% | 50 | 19/50 | 36 / 26 / 0 |
| `linear_call_atm` | +18.84% | +14.02% | 5 | 2/5 | 0 / 10 / 44 |
| `linear_call_otm` | +18.84% | +9.48% | 3 | 1/3 | 0 / 12 / 44 |
| `linear_direction_atm` | +18.84% | +14.02% | 5 | 2/5 | 0 / 21 / 86 |
| `linear_direction_otm` | +18.84% | +9.48% | 3 | 1/3 | 0 / 23 / 86 |

### Výnosy za kratší období

| Varianta | Nejlepší 7 dní | Nejlepší 30 dní | Nejlepší 90 dní | Zdvojnásobení 7 / 30 / 90 dní |
|---|---:|---:|---:|---|
| `options_off` | +11.46% | +26.97% | +29.52% | 0 / 0 / 0 |
| `reserved_cash` | +10.80% | +24.74% | +27.03% | 0 / 0 / 0 |
| `inverse_call_atm` | +10.24% | +26.63% | +31.95% | 0 / 0 / 0 |
| `inverse_call_otm` | +24.87% | +30.31% | +36.78% | 0 / 0 / 0 |
| `inverse_direction_atm` | +10.93% | +25.15% | +27.49% | 0 / 0 / 0 |
| `inverse_direction_otm` | +73.34% | +77.11% | +102.29% | 0 / 0 / 1 |
| `linear_call_atm` | +13.65% | +24.74% | +27.03% | 0 / 0 / 0 |
| `linear_call_otm` | +10.89% | +24.74% | +27.03% | 0 / 0 / 0 |
| `linear_direction_atm` | +13.65% | +24.74% | +27.03% | 0 / 0 / 0 |
| `linear_direction_otm` | +10.89% | +24.74% | +27.03% | 0 / 0 / 0 |

Okna se překrývají. Zisky a zdvojnásobení uvnitř doby držení závisí na modelových cenách opcí;
nejde o realizované výnosy ani budoucí pravděpodobnosti.

### Citlivost modelového propadu na IV

| Varianta | Poloviční IV | Vstupní IV | Dvojnásobná IV |
|---|---:|---:|---:|
| `inverse_call_atm` | -16.71% | -16.71% | -16.71% |
| `inverse_call_otm` | -18.54% | -18.28% | -17.64% |
| `inverse_direction_atm` | -17.69% | -17.69% | -17.69% |
| `inverse_direction_otm` | -19.65% | -18.38% | -18.34% |
| `linear_call_atm` | -17.57% | -17.52% | -17.52% |
| `linear_call_otm` | -17.63% | -17.52% | -17.52% |
| `linear_direction_atm` | -17.57% | -17.52% | -17.52% |
| `linear_direction_otm` | -17.63% | -17.52% | -17.52% |

Změna oceňovací IV nemění terminální opční cash flow ani zisk účtu; mění vykazovaný průběžný propad.

## Ověření a reprodukce

Účetně ověřeno 292 spotových/opčních cest; maximální NAV reziduum 2.55e-11. Vypnutý přepínač deleguje přímo na původní spotový engine.
Dřívější spotová čísla pro účet 1 000 se kontrolují automaticky. Hotovost a opční množství zůstávají nezáporné.
S kontrolními součty bylo ověřeno 39 zdrojových souborů a 550 historických obchodních oken; zapsané obchody i delivery ceny odpovídají surovým oficiálním odpovědím.
Měsíční expirace odstraňuje otevřené opce před konečným vyčíslením. Součet P&L obou oddělených částí tvoří P&L účtu.

Uzavřená kniha obsahuje 105 modelových nákupů napříč variantami. Samostatný součet prémií, poplatků a čistých výplat i součet P&L odpovídají výsledkům; nejvyšší reziduum 3.64e-12.

```bash
python -m scripts.download_long_options
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m scripts.research_long_options --options-enabled
python -m scripts.write_long_options_report
python -m pytest -q tests
```

Bez `--options-enabled` výzkumný příkaz porovnává pouze spot. Výzkum nemá napojení na objednávky.
[THETABOT_LONG_OPTIONS_2026-10-10.json](THETABOT_LONG_OPTIONS_2026-10-10.json) obsahuje všechny modelové výsledky, diagnostiku, hash zdrojových pozorování a protokolu.
[THETABOT_LONG_OPTIONS_2026-10-10_CLOSED_OPTIONS.csv](THETABOT_LONG_OPTIONS_2026-10-10_CLOSED_OPTIONS.csv) obsahuje každý uzavřený nákup: cenu, poplatky, výplatu, ID a čas zdrojového obchodu.
Evidence SHA-256 `13163da69f925a7ea587859da54f4521ef29f78193f05497f5f5c6bb5b89adbb`.
Surové API odpovědi a každý vybraný obchod mají URL, parametry, ID/čas a hash v `data/raw/long_options/`.
Účetní knihy a 4h křivky jsou reprodukovatelné v `artifacts/long_options_20261010/`.

## Omezení výsledku

- Trade prices are not historical executable bid/ask quotes or market depth.
- Constant-entry-IV Black-Scholes held values are synthetic, including model drawdowns.
- Uniform current fee scenarios and conservative lot floors are not dated rule histories.
- Binance spot converts inverse premiums/payoffs; index/venue basis remains.
- USD/USDT/USDC parity, stablecoin and exchange failure risk are not simulated.
- These spot dates and momentum signals were studied before; not an unseen holdout.
- A fixed 10% initial option sleeve is not replenished; wins can change its later weight.
- A 20% historical model drawdown is not a future loss guarantee.

Primární specifikace a zdroje: viz [protokol](LONG_OPTIONS_RESEARCH_PLAN.md).
