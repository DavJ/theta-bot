zvlastni je ze predikce pri horizontu 1 je jenom o neco malo lepsi nez raw a 0.5. Naopak delsi horizonty casto prinasi lepsi predikci. Mam pro to jen 2 vysvetleni 1) nejaka data z budoucna tedy 'horizontu' byla pouzita pro uceni i evaluaci metrik 2) theta model opravdu funguje a predikuje budoucnost trhu a tvar jeho chaotickeho atraktoru do budoucnosti

Ano, to je velmi zajímavé pozorování a obě tvé hypotézy jsou relevantní. Pojďme je rozebrat v kontextu opraveného skriptu (theta_eval_hbatch_jacobi_fixed_leak.py, File 1), u kterého jsme naposledy potvrdili, že je kauzálně čistý.

1) Únik dat (Look-ahead bias)?
Na základě naší poslední, velmi pečlivé kontroly kódu theta_eval_hbatch_jacobi_fixed_leak.py (File 1) jsme dospěli k závěru, že explicitní únik dat tam není. Model se v každém kroku t0 trénuje na datech striktně do t0-1 a predikuje pomocí vstupu z t0.

ALE: Tvoje pozorování je velmi důležité. Je možné, že i bez explicitního úniku může delší horizont (horizon=8) vykazovat lepší výsledky v tomto konkrétním testovacím režimu (walk-forward) z jiných důvodů:

Hladší cíl: Cíl y_target[t] = closes[t+horizon] - closes[t] pro horizon=8 je průměrná změna za 8 hodin. Pro horizon=1 je to změna za 1 hodinu. Cíl pro H=8 je mnohem "hladší" a méně zašuměný než pro H=1. Pro regresní model je snazší naučit se predikovat hladší signál.

Perzistence trendů: Trendy (nebo oscilace zachycené theta funkcemi) mohou být na 8hodinovém horizontu stabilnější a lépe extrapolovatelné než na 1hodinovém, kde převládá náhodný šum.

"In-sample contamination" (jemnější forma): I když model netrénuje na úplně stejném bodě, který testuje, trénuje na datech, která jsou jen o jeden krok pozadu. Pokud je signál velmi perzistentní (má vysokou autokorelaci), může model v tomto walk-forward testu vypadat lépe, než jaký by byl jeho výkon na zcela oddělených datech.

Závěr k hypotéze 1: Přímý únik dat tam není, ale charakter walk-forward testování může stále mírně nadhodnocovat výkon, potenciálně více u hladších, dlouhodobějších cílů.

2) Theta model opravdu funguje lépe na delší horizonty?
Toto je ta velmi zajímavá a teoreticky podložená hypotéza.

Teorie: Jak jsi zmínil, pokud předpokládáme, že theta funkce jsou řešením Fokker-Planckovy rovnice popisující dynamiku trhu, pak by měly zachycovat spíše základní "tvar" chaotického atraktoru než vysokofrekvenční šum.

Důsledek: Takový model by měl být přirozeně lepší v predikci delších, systematičtějších pohybů, kde se projeví tato základní dynamika, a naopak horší v predikci krátkodobého šumu (H=1), který je z definice spíše náhodný.

Tvé výsledky: Tvá data (vysoký výkon pro H=8, nízký pro H=1) jsou v souladu s touto teorií.

Závěr k hypotéze 2: Je to velmi plausibilní vysvětlení. Pokud tvůj model skutečně aproximuje řešení Fokker-Planckovy rovnice, pak je logické, že bude lépe fungovat na horizontech, kde se tato základní dynamika stihne projevit a není zcela přebita krátkodobým šumem.

Jak to definitivně rozhodnout?
Jediný způsob, jak spolehlivě odlišit "skutečnou dlouhodobou predikční sílu" (hypotéza 2) od "artefaktu walk-forward testování na hladkém cíli" (jemná forma hypotézy 1), je provést ten striktní Out-of-Sample (OOS) test:

Trénuj model POUZE na prvních 70 % dat.

Zmraz model (beta).

Testuj tento JEDEN fixní model na zbylých 30 % dat.

Pokud i v tomto testu bude výkon pro horizon=8 výrazně lepší než pro horizon=1 (a stále vysoký, např. R>0.3), pak jsi s vysokou pravděpodobností potvrdil hypotézu 2. Máš model, který skutečně zachycuje a extrapoluje základní dynamiku trhu na delších horizontech. ✨
