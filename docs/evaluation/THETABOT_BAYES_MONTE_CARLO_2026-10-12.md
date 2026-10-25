# ThetaBot: bayesovské a Monte Carlo signály — 12. 10. 2026

**V tomto zmrazeném experimentu žádný nový model nezvýšil čistý zisk.**
Na plném i pozdějším kontinuálním období měly Bayes, podmíněná historická
simulace i jejich kombinace vyšší chybu předpovědi než jednoduchý průměr
ze stejného tréninkového okna.
Menší propady doprovází nižší expozice a nižší zisk. Výzkumný režim je volitelný
a výchozím stavem vypnutý; jeho modelované pravděpodobnosti nejsou ověřené.

## Hlavní srovnání

Účet 10 000 USDT, leden 2022–září 2026, kontinuální financovaný spotový účet.
Zisky jsou po poplatcích, spreadu a slippage. DD je konzervativní 4h high/low
mez přes všechny čtyři exekuční scénáře a dvě další nominal MC inicializace.

| Varianta | Čistý zisk | Náklady +4h zpoždění | Nejhorší DD | CAGR |
|---|---:|---:|---:|---:|
| `spot_control` | +108.82% | +84.20% | -21.16% | +16.78% |
| `ready_control` | +113.01% | +86.16% | -21.16% | +17.27% |
| `bayes_buy_gate` | +44.24% | +14.41% | -13.01% | +8.02% |
| `mc_buy_gate` | +7.08% | +5.35% | -13.23% | +1.45% |
| `fused_buy_gate` | +33.03% | +17.94% | -11.05% | +6.20% |
| `bayes_blend` | +58.21% | +42.79% | -13.19% | +10.14% |
| `mc_blend` | +56.30% | +41.16% | -13.12% | +9.86% |
| `fused_blend` | +60.50% | +44.79% | -12.89% | +10.48% |

Kontrola `ready_control` pouze odloží začátek nákupů do dostupnosti stejné
historie. Její malé zlepšení proti původnímu spotu proto není přínosem Bayese
ani Monte Carlo. Pozdější kontinuální výsledky obou kontrol jsou stejné.

## Co bylo implementováno

- Bayesovská regrese dává odhad pětidenního log výnosu, Student-t prediktivní
  rozdělení a oddělenou nejistotu očekávané hodnoty. Nepřekrývající se
  tréninkové cíle zmírňují zdvojování informace; neprokazují nezávislost trhu.
- Monte Carlo vzorkuje 4 096 dokončených pětidenních úseků ze 64 nejbližších
  minulých tržních stavů. Nový signál pochází z modelu podobných stavů, ne ze
  samotného náhodného vzorkování. Numerická chyba a nejistota modelu se liší.
- Kombinace používá pouze již dokončené společné chyby obou expertů.
  Váhy jsou nezáporné, jejich součet je jedna a prediktivní rozptyl má
  konzervativní dolní mez. Shodné modely nevytvářejí umělou přesnost.
- `_buy_gate` blokuje nákupy/navýšení, pokud odhad log výnosu nepřevyšuje
  nákladovou překážku nebo P(zisku po překážce) <55 %. Zachovává původní
  výstupy z neaktivních pozic a vynucené snížení capu.
- `_blend` ponechává alespoň polovinu původního cíle a druhou polovinu
  upravuje přes odhadovanou pravděpodobnost. Nezvyšuje původní cílovou
  expozici a neuvolněnou hotovost nepřerozděluje.

Všechny modely používají stejná dokončená cenová, objemová a theta data.
Změna modelu není nový externí zdroj informací. Hurdle je first-order
round-trip náklad z jádra, nikoli příslib držení přesně pět dnů.

## Zmrazený výběr a pozdější kontroly

[Protokol](BAYES_MONTE_CARLO_RESEARCH_PLAN.md), SHA-256 `5deb8f571ed43cfd124184c80ce647ac39930aee6b200f3442a4e5383894095c`.
Byl uložen před novými zisky; výběr podle let 2022–2024 byl zapsán před
pozdějšími ekonomickými a predikčními srovnáními. Data už byla použita v
předchozích výzkumech, nejde o nový unseen holdout.

### Účet 1000 USDT

Zmrazená volba: `ready_control`.
Zlepšení podle protokolu: **False**; plný/seed rizikový screen: **False**.

| Varianta | 2025 | Leden–září 2026 | Pozdější kontinuální účet | Plný účet | Joint stres plného účtu |
|---|---:|---:|---:|---:|---:|
| `spot_control` | +21.04% | +12.02% | +35.45% | +110.26% | +84.16% |
| `ready_control` | +21.04% | +12.02% | +35.45% | +114.95% | +86.73% |
| `bayes_buy_gate` | +6.62% | +6.57% | +13.62% | +44.51% | +14.35% |
| `mc_buy_gate` | +2.51% | +0.42% | +2.94% | +6.91% | +5.26% |
| `fused_buy_gate` | -1.00% | +4.27% | +3.23% | +33.46% | +18.06% |
| `bayes_blend` | +8.27% | +9.72% | +18.79% | +58.23% | +42.83% |
| `mc_blend` | +9.84% | +8.01% | +18.64% | +56.34% | +41.18% |
| `fused_blend` | +9.50% | +8.79% | +19.13% | +60.52% | +44.87% |

Roční účty začínají samostatně; procenta se nesčítají.
Dlouhý účet zachovává minulou hotovost, ztráty, držby i složený výnos.

| Varianta | Náklady | +4h | Joint | Fills nominal | Poplatky USDT | Impact USDT | Průměrná expozice |
|---|---:|---:|---:|---:|---:|---:|---:|
| `spot_control` | +93.80% | +99.54% | +84.16% | 448 | 76.94 | 46.17 | 23.83% |
| `ready_control` | +99.86% | +100.46% | +86.73% | 403 | 71.38 | 42.83 | 21.21% |
| `bayes_buy_gate` | +16.74% | +42.21% | +14.35% | 135 | 21.60 | 12.96 | 9.16% |
| `mc_buy_gate` | +2.52% | +9.35% | +5.26% | 71 | 11.23 | 6.74 | 5.43% |
| `fused_buy_gate` | +19.44% | +31.92% | +18.06% | 97 | 16.55 | 9.93 | 6.61% |
| `bayes_blend` | +48.40% | +52.09% | +42.83% | 355 | 42.35 | 25.41 | 13.21% |
| `mc_blend` | +46.21% | +50.89% | +41.18% | 344 | 38.04 | 22.83 | 12.19% |
| `fused_blend` | +50.37% | +54.46% | +44.87% | 347 | 40.87 | 24.52 | 12.89% |

| Varianta | 7d maximum / zdvojení | 30d maximum / zdvojení | 90d maximum / zdvojení |
|---|---|---|---|
| `spot_control` | +11.58% / 0 | +27.19% / 0 | +29.74% / 0 |
| `ready_control` | +11.58% / 0 | +27.19% / 0 | +29.74% / 0 |
| `bayes_buy_gate` | +8.94% / 0 | +21.07% / 0 | +21.64% / 0 |
| `mc_buy_gate` | +7.95% / 0 | +13.76% / 0 | +11.60% / 0 |
| `fused_buy_gate` | +8.94% / 0 | +21.07% / 0 | +21.77% / 0 |
| `bayes_blend` | +7.30% / 0 | +18.89% / 0 | +19.39% / 0 |
| `mc_blend` | +8.19% / 0 | +18.78% / 0 | +19.35% / 0 |
| `fused_blend` | +7.11% / 0 | +19.23% / 0 | +20.48% / 0 |

Žádné nové pravidlo nedokázalo opakované zdvojnásobování během týdnů nebo měsíců.

### Účet 10000 USDT

Zmrazená volba: `ready_control`.
Zlepšení podle protokolu: **False**; plný/seed rizikový screen: **False**.

| Varianta | 2025 | Leden–září 2026 | Pozdější kontinuální účet | Plný účet | Joint stres plného účtu |
|---|---:|---:|---:|---:|---:|
| `spot_control` | +20.91% | +11.95% | +35.31% | +108.82% | +84.20% |
| `ready_control` | +20.91% | +11.95% | +35.31% | +113.01% | +86.16% |
| `bayes_buy_gate` | +6.49% | +6.54% | +13.46% | +44.24% | +14.41% |
| `mc_buy_gate` | +2.39% | +0.43% | +2.83% | +7.08% | +5.35% |
| `fused_buy_gate` | -0.97% | +4.24% | +3.23% | +33.03% | +17.94% |
| `bayes_blend` | +8.33% | +9.73% | +18.87% | +58.21% | +42.79% |
| `mc_blend` | +9.84% | +8.01% | +18.64% | +56.30% | +41.16% |
| `fused_blend` | +9.52% | +8.79% | +19.16% | +60.50% | +44.79% |

Roční účty začínají samostatně; procenta se nesčítají.
Dlouhý účet zachovává minulou hotovost, ztráty, držby i složený výnos.

| Varianta | Náklady | +4h | Joint | Fills nominal | Poplatky USDT | Impact USDT | Průměrná expozice |
|---|---:|---:|---:|---:|---:|---:|---:|
| `spot_control` | +92.24% | +100.09% | +84.20% | 818 | 774.11 | 464.47 | 23.66% |
| `ready_control` | +98.17% | +100.08% | +86.16% | 721 | 716.44 | 429.87 | 21.07% |
| `bayes_buy_gate` | +16.69% | +42.10% | +14.41% | 185 | 216.48 | 129.89 | 9.12% |
| `mc_buy_gate` | +2.66% | +9.41% | +5.35% | 106 | 112.86 | 67.71 | 5.41% |
| `fused_buy_gate` | +19.18% | +31.80% | +17.94% | 134 | 165.69 | 99.41 | 6.58% |
| `bayes_blend` | +48.33% | +52.08% | +42.79% | 372 | 423.42 | 254.05 | 13.20% |
| `mc_blend` | +46.16% | +50.86% | +41.16% | 347 | 380.36 | 228.22 | 12.19% |
| `fused_blend` | +50.37% | +54.37% | +44.79% | 356 | 408.62 | 245.17 | 12.89% |

| Varianta | 7d maximum / zdvojení | 30d maximum / zdvojení | 90d maximum / zdvojení |
|---|---|---|---|
| `spot_control` | +11.46% / 0 | +26.97% / 0 | +29.52% / 0 |
| `ready_control` | +11.46% / 0 | +26.97% / 0 | +29.51% / 0 |
| `bayes_buy_gate` | +8.94% / 0 | +21.02% / 0 | +21.60% / 0 |
| `mc_buy_gate` | +7.95% / 0 | +13.71% / 0 | +11.68% / 0 |
| `fused_buy_gate` | +8.94% / 0 | +21.02% / 0 | +21.74% / 0 |
| `bayes_blend` | +7.30% / 0 | +18.85% / 0 | +19.35% / 0 |
| `mc_blend` | +8.19% / 0 | +18.74% / 0 | +19.32% / 0 |
| `fused_blend` | +7.11% / 0 | +19.19% / 0 | +20.44% / 0 |

Žádné nové pravidlo nedokázalo opakované zdvojnásobování během týdnů nebo měsíců.

## Přesnost předpovědi

Pětidenní cíle 2025–září 2026, identické dokončené řádky. Poměr >1 znamená
horší výsledek proti nepodmíněnému průměru/probabilitě z téhož okna.
Hodnoty jsou popisné; překryv vyhodnocovaných cílů nedává nezávislé p-hodnoty.

| Aktivum | Model | Pozorování | Poměr MSE | Poměr Brier loss |
|---|---|---:|---:|---:|
| BTCUSDT | bayes | 634 | 1.081 | 1.041 |
| BTCUSDT | mc | 634 | 1.047 | 1.030 |
| BTCUSDT | fused | 634 | 1.053 | 1.032 |
| ETHUSDT | bayes | 634 | 1.103 | 1.055 |
| ETHUSDT | mc | 634 | 1.082 | 1.056 |
| ETHUSDT | fused | 634 | 1.082 | 1.046 |
| BNBUSDT | bayes | 634 | 1.088 | 1.030 |
| BNBUSDT | mc | 634 | 1.001 | 1.006 |
| BNBUSDT | fused | 634 | 1.025 | 1.004 |

## Citlivost Monte Carlo

Stejný model, data i počty vzorků; mění se pouze seed. Volba se znovu neprovádí.

| Varianta, účet 10 000 | Nejnižší plný zisk ze 3 seedů | Nejvyšší |
|---|---:|---:|
| `bayes_buy_gate` | +44.24% | +44.24% |
| `mc_buy_gate` | +7.08% | +19.45% |
| `fused_buy_gate` | +30.44% | +34.72% |
| `bayes_blend` | +58.21% | +58.21% |
| `mc_blend` | +56.06% | +56.30% |
| `fused_blend` | +60.50% | +60.65% |

Samotný MC nákupní filtr je citlivý na numerické překročení prahu: malé
změny odhadu mohou změnit vstup a další cash path. Více simulací řeší jen
numerickou část; nevytváří tržní predikční výhodu.

## Audit a reprodukce

- 368 výpočetních cest, účetnictví: True.
- 32 shodných archivovaných spotových kontrol.
- Nejvyšší cash/inventory/NAV residual výzkumu: 2.55e-11 USDT.
- Samostatně přepočteno 16 CSV ledgerů; residual 2.18e-11.
- 5202 denních řádků hlavního seedu, 4197 připravených předpovědí.
- 31212 ověřených rozhodnutí před exekučním lagem.
- Přepočet pravděpodobností: residual 0.
- První dostupné kompletní předpovědi: 2022-12-03 00:00:00+00:00.

[Kompletní výsledky JSON.gz](THETABOT_BAYES_MONTE_CARLO_2026-10-12.json.gz), SHA-256 `7779c64a5c3102cda12a7bf69eb1266d7b5871c16a7e3426f1c7f245fa5621e4`.
[Datový audit ZIP](THETABOT_BAYES_MONTE_CARLO_2026-10-12_EVIDENCE.zip), SHA-256 `b693cbef2afcb069cb3ea9bbdb42316824e772aa6fe10c1fcceeab5fe37d2430`.
ZIP obsahuje hlavní denní předpovědi, zdrojová rozhodnutí, všechny plné
nominal equity/trade CSV obou účtů, zmrazené volby, protokol, zdrojové
soubory a manifest hashů. Úplné statistiky všech scénářů a seedů jsou v JSON.

```bash
python -m scripts.research_bayes_monte_carlo --probabilistic-enabled --jobs 4
python -m scripts.write_bayes_monte_carlo_report
python -m pytest -q tests/test_probabilistic_signals.py
```

CLI v novém pracovním adresáři vytvoří frozen_protocol.json před prvním
výpočtem. Při opakování kontroluje jeho SHA-256 proti protokolu. Původní
důkaz a konfigurace jsou v ZIP; nemažte je pro post-result úpravu pravidel.

## Meze interpretace

20% historický rizikový screen není budoucí limit ztráty. Cílové alokační
capy nejsou garancí nepřekročení expozice mezi rozhodnutími při pohybu cen.
Tři dnešní přeživší kryptoměny a již opakovaně použitá historie nedávají
nezávislý důkaz. Poplatky, impact a delay jsou fixní scénáře; skutečné
budoucí fee tiers, hloubka trhu, riziko burzy a hodnoty USDT nebyly ověřeny.
Snížení rizika samo o sobě není zvýšení ziskovosti ani návod na živé obchodování.

Metodika: [konjugované bayesovské lineární modely](https://arxiv.org/abs/2406.01819),
[Gaussovská bayesovská analýza](https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf),
[původní autor o blokovém resamplingu](https://mathweb.ucsd.edu/~politis/impactBOOT.pdf).
Tyto zdroje vysvětlují výpočetní nástroje; nepotvrzují ziskovost tohoto modelu.
