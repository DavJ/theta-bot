# Reliability, Profitability & Risk by Roadmap Phase

> Odhady jsou orientační, kalibrované pro krypto páry typu BTCUSDT/ETHUSDT, timeframe 5–15m, s realistickými poplatky a mírným slippage. U malého kapitálu (≈1 000 USD) očekávej spíše spodní hranice rozpětí.

## Přehledová tabulka

| Fáze | Strategie / modul | Odhad spolehlivosti signálů | Očekávaná výdělečnost* | Rizikovost (DD, whipsaw, margin) | Inovativní prvek vs. běžní boti |
|------|-------------------|-----------------------------|------------------------|----------------------------------|----------------------------------|
| **1. MVP (MVA)** | Klouzavé průměry | 45–55 % | ±0 % až +5 % / měsíc (jen v trendu) | Vysoká – whipsaw v range | Žádný, jen sandbox |
| **2. Predikce (Theta + Kalman + NN)** | Theta rezidua → Kalman → LSTM/RNN | 52–60 % (víc v trendu) | +5–15 % / měsíc po poplatcích | Střední – model drift, občasný šum | **Theta transformace** + **Kalman smoothing** (unikátní preprocessing) |
| **3. Hedging / Pair / Arbitráž** | Párové obchody, hedge β | ~stejné hit-rate; nižší DD | +3–10 % / měsíc, stabilnější | Nižší – hedge zplošťuje ztráty; riziko spreadu | **Entanglované obchody** (systematicky párované vstupy) |
| **4. Options-based Hedging** | Spot + opce (protective puts/calls) | ~stejně; PnL stabilnější | +2–8 % / měsíc po prémiích | Nízká–Střední – hlídat margin/funding | Automatizace opčního hedgingu nad AI signálem |
| **5. Cloud & Multi-strategy** | Ensemble: predikce + hedge + arbitráž, více párů | 55–65 % (ensemble efekt) | +10–25 % / měsíc při škálování | Střední – korelace mezi páry, tech rizika | **Hybridní AI + kvantová matematika** (theta, FPE, bikvaterniony) |

\* „Výdělečnost“ = orientační čistý PnL po poplatcích/slippage při optimálním risku a timeframe. Reálný výsledek závisí na likviditě, slippage, režimu trhu a disciplíně risk managementu.

---

## Proč je Fáze 2 „sweet spot“
- Rychle implementovatelná (theta + Kalman + jednoduchá NN), přitom poskytuje **nezávislý edge** vůči běžným TA botům.
- Při **+5 % / měsíc** složené úročení ⇒ ~**80 % ročně** (bez navyšování % risku na trade).  
- Snadno rozšířitelná o **hedge modul** (Fáze 3), který sníží DD a zvýší komfort při škálování kapitálu.

---

## Poznámky k metrice a kalibraci
- **Hit-rate** samo o sobě nestačí – důležitá je **PnL distribuce** (cut loss, let winners run).
- **Timeframe**: 5–15m bývá lepší kompromis mezi šumem a náklady než 1m.
- **Backtest**: používat **walk‑forward** (refit parametrů po blocích), simulovat poplatky, skluz a omezení burzy.
- **Hedge β**: kalibrovat OLS na log‑returnech (rolling), ne fixně.

---

## (Volitelné – po dokončení hlavních fází) Low‑risk profit enhancers
- Maker/limit **fee optimalizace** (snížení nákladů bez zvýšení rizika).
- **Stablecoin triangulární arbitráž** (USDT/USDC/BUSD) – drobné, ale téměř bezrizikové přísuny.
- **Funding rate harvesting** na perpetuals (long/short podle znaménka fundingu).
- **Cash & Carry** (spot + futures) – výnos z basis při zajištěné delta.
- **Dynamická β‑hedge kalibrace**, **volatility targeting**, **no‑trade zóny** (ATR/šumový filtr).
