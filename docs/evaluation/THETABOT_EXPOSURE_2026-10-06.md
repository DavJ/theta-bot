# ThetaBot: vyšší expozice, vyšší výnos i propad

Dosavadní 30% strop byl zachovaný výzkumný předpoklad, nikoli limit zadaný uživatelem. Nyní porovnáváme stejný momentum model při 30%, 50%, 75% a 100% stropu. Jde o nákup spotových aktiv z vlastní hotovosti. Žádná páka ani půjčka nejsou použity.

## Výsledek na již známých obdobích

| Strop cílové expozice | Čistý výnos 2025 | Čistý výnos leden–září 2026 | Souvislý propad 2022–2026 |
|---|---:|---:|---:|
| 30 % | +11.45% | +6.16% | -10.52% |
| 50 % | +19.59% | +10.18% | -16.98% |
| 75 % | +30.53% | +14.69% | -24.49% |
| 100 % | +41.64% | +19.07% | -31.34% |

Obě roční přehrávky začínají nezávisle s 1 000 USDT. Souvislý účet začíná s jedinými 1 000 USDT na začátku roku 2022 a není mezi roky resetován. Všechny obchody, kompozice portfolia a náklady se znovu počítají; výnosy nejsou násobkem staré tabulky.

## Jediný souvislý účet

| Strop | Konečný stav z 1 000 USDT | Celkový výnos za celé období | Historický roční přepočet | Průměrná expozice |
|---|---:|---:|---:|---:|
| 30 % | 1549.40 USDT | +54.94% | +9.66% | 14.13% |
| 50 % | 2039.10 USDT | +103.91% | +16.18% | 23.47% |
| 75 % | 2793.73 USDT | +179.37% | +24.14% | 35.20% |
| 100 % | 3723.57 USDT | +272.36% | +31.88% | 46.84% |

Roční přepočet je geometrický přepočet jedné historické cesty, nikoli předpověď budoucích výnosů.

## Nepříznivý rok a delší růstové období

| Strop | Čistý výnos 2022 | Propad 2022 | Souhrnný výnos 2023–2024 | Propad 2023–2024 |
|---|---:|---:|---:|---:|
| 30 % | -2.12% | -9.94% | +33.82% | -10.20% |
| 50 % | -4.11% | -16.28% | +61.40% | -16.55% |
| 75 % | -7.02% | -23.83% | +100.70% | -23.92% |
| 100 % | -10.61% | -31.00% | +146.43% | -30.73% |

Rok 2022 začíná bez historie před rokem 2022; prvních 60 dní model čeká na volatilitu. Nejde tedy o plně zahřátý test od prvního lednového dne. Výnos 2023–2024 je souhrn za dva roky, nikoli roční výnos.

## Dvojnásobné náklady

| Strop | Čistý výnos 2025 | Čistý výnos leden–září 2026 |
|---|---:|---:|
| 30 % | +10.59% | +5.24% |
| 50 % | +17.80% | +8.68% |
| 75 % | +28.15% | +12.11% |
| 100 % | +38.12% | +15.61% |

Běžné náklady: 0,1 % poplatek při každém provedení, 5 bazických bodů skluzu a polovina spreadu 2 bazické body. Nákladový stres zdvojnásobuje poplatek a skluz, spread nechává stejný. Minimální obchod je 10 USDT, běžné změny probíhají v pondělí s pásmem 1 % NAV. Výstupy do hotovosti a snížení překročené expozice se kontrolují denně. Limit na jedno aktivum je polovina celkového stropu.

## Co tento výsledek znamená

Vyšší kapitálová expozice zvedla výnos i hloubku propadů. Nevytvořila novou predikční výhodu. Momentum bylo vyzdviženo až po předchozím srovnání novějších dat, takže tyto výsledky nejsou nezávislým potvrzením jeho výběru. Všechna období jsou známá výzkumná data.

Dosavadní 15% limit vývojového propadu byl výzkumný předpoklad, nikoli známá uživatelova tolerance rizika. Tento experiment žádný limit automaticky neuvolňuje, nevybírá vítěze a nemění konfiguraci živého obchodování. Historický propad není záruka maximální budoucí ztráty.

Denní simulace vynechává latenci, fronty příkazů a intradenní průběh ceny. Konečné pozice jsou oceněny trhem, ne nuceně zlikvidovány. Dnešní výběr přeživších BTC/ETH/BNB nezaručuje nezaujatý výběr aktiv. Příjmy z hotovosti a daně nejsou modelovány.

Úplné náklady, obchody, metriky a manifesty oficiálních ověřených archivů jsou v doprovodném JSON. Protokol: [EXPOSURE_SENSITIVITY_PLAN.md](EXPOSURE_SENSITIVITY_PLAN.md).

## Opakování

```bash
python -m scripts.research_exposure --daily data/raw/*_1d_*.csv --report-out docs/evaluation
```
