# ThetaBot: výnos při rozpočtu propadu 20 %

Uživatel zvýšil přijatelný dočasný propad účtu z 10 % na **20 %**. Pevné srovnání přidává škálování podle kovariance a podle zbývajícího rozpočtu ztráty. Momentum signály jsou stejné jako v předchozím výzkumu; nepřidáváme páku.

**Zmrazený výběr z let 2022–2024: `static_50`. Historický screening: prošel.** Výběr maximalizuje čistý výnos z vývoje pod 20% konzervativní mezí propadu. Všechna období už byla známá, proto nejde o nové nezávislé potvrzení výhody.

## Výsledky po nákladech

| Varianta | Výnos 2025 | Výnos leden–září 2026 | Souvislý výnos 2022–2026 | Souvislý propad ze závěrů | Konzervativní intradenní mez propadu | Všechny historické propady do 20 % |
|---|---:|---:|---:|---:|---:|:---:|
| static_50 (zmrazeno) | +19.59% | +10.18% | +103.91% | -16.98% | -18.54% | ano |
| static_75 | +30.53% | +14.69% | +179.37% | -24.49% | -26.59% | ne |
| vol_15 | +18.84% | +11.05% | +89.59% | -12.46% | -13.52% | ano |
| vol_25 | +32.81% | +16.74% | +171.39% | -19.94% | -21.93% | ne |
| cushion_3 | +19.76% | +5.12% | +54.72% | -13.91% | -14.95% | ano |
| cushion_5 | +30.06% | +4.50% | -1.27% | -16.43% | -18.29% | ano |
| vol_25_cushion_5 | +32.34% | +10.05% | +62.06% | -15.21% | -17.23% | ano |

Roční přehrávky začínají každá s 1 000 USDT; souvislá přehrávka používá jediný účet s počátečními 1 000 USDT a nikdy neresetuje hotovost ani rizikové maximum. Intradenní mez kombinuje denní maxima a minima držených aktiv; nemusely nastat zároveň ani v pořadí maximum–minimum. Jde o konzervativní mez, nikoli přesně naměřený propad.

## Souvislý účet a nákladový stres

| Varianta | Stav z 1 000 USDT | Historický roční přepočet | Průměrná expozice | Výnos při dvojnásobných nákladech | Propadová mez při dvojnásobných nákladech |
|---|---:|---:|---:|---:|---:|
| static_50 | 2039.10 USDT | +16.18% | 23.47% | +88.24% | -19.24% |
| static_75 | 2793.73 USDT | +24.14% | 35.20% | +148.12% | -27.54% |
| vol_15 | 1895.88 USDT | +14.41% | 18.39% | +77.30% | -14.19% |
| vol_25 | 2713.88 USDT | +23.39% | 29.93% | +144.08% | -23.51% |
| cushion_3 | 1547.17 USDT | +9.62% | 16.36% | +39.31% | -15.22% |
| cushion_5 | 987.35 USDT | -0.27% | 2.99% | +0.16% | -18.79% |
| vol_25_cushion_5 | 1620.64 USDT | +10.70% | 18.51% | +37.54% | -17.23% |

Roční přepočet popisuje jednu historickou cestu; není předpovědí výnosu. Běžný simulovaný náklad je poplatek 0,1 % při každém provedení, skluz 5 bazických bodů a spread 2 bazické body (polovina při každém provedení). Stres zdvojnásobuje poplatek a skluz. Náklady jsou účtované za skutečné obchody se společnou hotovostí, nikoli odečtené paušálně.

## Zmrazený výběr a kontrolní podmínky

| Podmínka | Výsledek |
|---|:---:|
| development_eligible | splněno |
| historical_drawdown_limit | splněno |
| later_nominal_positive | splněno |
| later_and_continuous_cost_stress_positive | splněno |

Vývojový čistý výnos `static_50`: +54.33%; konzervativní mez propadu -17.89%. Volba byla zapsaná před výpočtem novějších období; jiného pozdějšího vítěze automaticky nenasazujeme.

Nákladový stres vybrané varianty: 2025 +17.80%, leden–září 2026 +8.68%.

## Umělý společný cenový skok −40 %

Předem zvolený scénář násobí ceny všech tří aktiv od otevření 11. března 2024 faktorem 0,6. Následující relativní pohyby zachovává, signály i řízení znovu počítá. Jde o zátěžovou konstrukci, nikoli další pozorovaný historický výsledek. Test ukazuje, zda otevřený cenový skok může porušit 20% rozpočet ještě před prodejem.

| Varianta | Souvislý výnos ve scénáři | Propad ze závěrů | Konzervativní mez propadu |
|---|---:|---:|---:|
| static_50 | +67.13% | -26.02% | -27.30% |
| static_75 | +102.73% | -38.10% | -39.71% |
| vol_15 | +63.98% | -20.27% | -21.34% |
| vol_25 | +113.57% | -31.10% | -32.60% |
| cushion_3 | +6.94% | -19.09% | -19.75% |
| cushion_5 | -1.27% | -16.43% | -18.29% |
| vol_25_cushion_5 | -1.09% | -24.55% | -25.34% |

## Pravidla a omezení

`vol_15` a `vol_25` používají 60 uzavřených denních výnosů a kovarianci staženou o 25 % k diagonále; cíle jsou 15 % a 25 % roční volatility portfolia. `cushion_3` a `cushion_5` omezují investovanou část na 3× nebo 5× zbývající odstup od podlahy 82 % historického maxima účtu. Kombinace respektuje oba limity. Rizikové maximum zahrnuje konzervativní cenovou mez předchozích uzavřených dní. Aktuální denní rozsah ceny nesmí ovlivnit příkazy na jeho otevření.

Běžné nákupy a přesuny jsou v pondělí s pásmem 1 % NAV. Výstup neaktivního signálu a rizikové snížení probíhají na nejbližším denním otevření; rizikové snížení obchází toto pásmo. Minimum 10 USDT zůstává platné a může ponechat malý zbytek pozice. Podlaha ani maximum se po ztrátě neresetují; účet pod podlahou může zůstat v hotovosti bez možnosti obnovit výnos. Četnější rizikové redukce mohou zvýšit náklady.

20% hranice je kritérium výběru. Denní řízení ani prodejní příkaz nezaručí její dodržení při budoucím skoku ceny, omezené likviditě nebo zpoždění. Žádná varianta není tímto výpočtem zapnutá pro živé obchodování. Otevřené konečné pozice jsou oceněné trhem; konečná nucená likvidace a její náklady, úročení hotovosti, daně a skutečná likvidita nejsou modelované.

Všechny uložené cesty byly účetně zkontrolovány: nezáporná hotovost a pozice, NAV ze skutečného ocenění pozic, poplatky a cenový dopad souhlasí s obchody. To ověřuje provedení simulace, nikoli budoucí výdělečnost.

[Předem zapsaný protokol](RISK_BUDGET_RESEARCH_PLAN.md); doprovodný JSON obsahuje metriky všech přehrávek, rizikové nastavení, kontrolní podmínky a zdrojové manifesty.

Inspirace: [Moreira a Muir, Volatility Managed Portfolios](https://www.nber.org/papers/w22208). Omezení provedení stop příkazů: [SEC / Investor.gov](https://www.investor.gov/introduction-investing/general-resources/news-alerts/alerts-bulletins/investor-bulletins-14). Tyto zdroje nepotvrzují výdělečnost tohoto krypto modelu.

## Opakování

```bash
python -m scripts.research_risk_budget --daily data/raw/*_1d_*.csv --report-out docs/evaluation
```
