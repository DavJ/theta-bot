Ověření kauzální čistoty skriptu theta_eval_hbatch_jacobi_fixed_leak.py
1. Trénink pouze na minulých datech (bez look-ahead v čase t0)

Skript provádí walk-forward evaluaci, kde se model v každém kroku t0 trénuje jen na historických datech do t0-1 a poté predikuje hodnotu v čase t0 (o horizon dopředu). V kódu je to zajištěno nastavením hi_train = t0, takže tréninkové okno zahrnuje indexy [lo_train, ..., t0-1] a nekouká za predikovaný bod. Poslední tréninkový cíl y_target[t0-1] využívá cenu closes[(t0-1)+horizon], což je stále před cenou closes[t0+horizon] predikovanou modelem. Tím je zaručeno, že při predikci v čase t0 model nevidí žádná data, která by pocházela z budoucnosti tohoto bodu.

2. Žádné budoucí ceny neprosakují do tréninku ani parametrů

Při výpočtu cílové proměnné není do tréninku zahrnuta žádná skutečná budoucí cena typu close[t0+h]. Cílové hodnoty (y_target) jsou sice definovány jako rozdíl ceny za horizon (např. y_target[i] = close[i+h] - close[i]), avšak v každém kroku se pro trénink berou pouze ty cíle, jejichž budoucí cena leží před predikovaným horizontem. Například pro predikci v čase t0 s horizontem h je nejvzdálenější cenou použitou v tréninku closes[t0-1+h], což je o jeden krok méně než closes[t0+h]. Tím pádem se žádná cena za horizont predikce do výběru parametrů ani učení nedostane. Model také neprovádí žádnou ladící selekci parametrů na základě budoucí výkonnosti – volba parametrů je buď fixní, nebo vychází z historických dat a hyperparametrů zadaných dopředu.

3. Váhové funkce a transformace featur jsou odvozené jen z parametrů a historie

Skript využívá theta bázi a váhování, které nečerpají z budoucích hodnot cíle ani cen:

Funkce build_theta_q_basis(...) generuje matici příznaků (theta-funkcí) čistě z časových indexů a zadaných parametrů (baseP, sigma, N_even, N_odd). To znamená, že featury závisí jen na čase a parametrech, nikoliv na budoucích cenách.

Váhy q_weights (Jacobiho $q$-váhy) jsou počítány uvnitř této báze jen z parametrů a indexů, a time_weights jsou určeny exponenciálním oknem podle délky tréninkového úseku. Ani jedny tedy nevycházejí z budoucích dat – time_weights se odvozují od velikosti aktuálního okna (počtu historických vzorků) a nastaveného $\alpha$, a q_weights jen z parametrů modelu.

Skript nepoužívá žádnou funkci typu _build_features pro škálování či standardizaci na základě celých dat; tím se vyhýbá riziku „úniku“ globálních statistických informací z budoucnosti. Všechny transformace vstupů jsou tedy kauzálně bezpečné.

Navíc je predikce realizována průměrováním (ensemble avg) několika modelů, nikoli výběrem ex-post nejlepší predikce. I kdyby byl použit režim max, výběr k nejlepších komponent by závisel jen na parametrech $\beta$ natrénovaných na minulosti.

4. Korektní ukládání predikcí a kauzální výpočet metrik

Každá predikce je uložena se správným časovým označením a následně porovnána se skutečností způsobem zaručujícím kauzální čistotu. Ve smyčce predikcí se pro každý bod ukládají záznamy obsahující např. entry_idx = t0 (čas, kdy byla predikce učiněna) a compare_idx = t0 + horizon (čas, ke kterému se vztahuje budoucí skutečná cena). Spolu s tím se uchovává vypočtená pred_delta a odpovídající true_delta pro daný časový bod. Tím je zajištěno, že predikce a skutečnost jsou správně spárovány v čase.

Na konci skript pomocí funkce metrics(...) propočte metriky výkonu pouze na základě těchto uložených predikcí a skutečných změn. Korelace corr_pred_true je tedy počítána pouze z dvojic (predikovaný rozdíl, skutečný rozdíl) pro stejné časové indexy, které model neviděl při tréninku. Stejně tak hit_rate_pred (úspěšnost predikce směru pohybu) porovnává predikovaný směr s realizovaným směrem pro každý bod ex-post, takže reflektuje kauzálně správný výkon modelu. Jinými slovy, metriky jsou vyhodnoceny až po dokončení všech predikcí, a žádný budoucí výsledek neovlivňuje dřívější predikce ani výpočet těchto metrik.

Verdikt kauzality: Po pečlivé kontrole lze říci, že skript je z hlediska časového úniku dat kauzálně čistý – model v každém kroku používá jen informace dostupné do daného času. Nenašli jsme žádný bod, kde by se přímo míchala budoucí skutečnost do tréninku či predikce. (Jediným možným zdrojem mírného nadhodnocení může být charakter walk-forward testu na vysoce autokorelovaných datech – model je vždy trénován jen o krok pozadu, což může zlepšit krátkodobou extrapolaci trendů. To však není explicitní únik dat, spíše metodologický artefakt. Pro úplnost výkon ověřujeme následujícími testy.)

Návrh sanity testů pro potvrzení integrity modelu

Abychom definitivně vyloučili jakýkoliv skrytý leakage a ověřili, že výsledek modelu není artefakt, je vhodné provést dodatečné sanity testy:

Shuffle Test (náhodné promíchání): Náhodně permutuj vektor skutečných změn (true_delta) mezi sebou a znovu spočítej korelaci s původními predikcemi (pred_delta). Pokud byl model kauzálně v pořádku, korelace pred_delta s promíchanou pravdou by měla spadnout k nule. Tím ověříme, že původní vysoká korelace nevznikla nějakým triviálním posunem či systémovou chybou.

Lag Test (časový posun): Posuň skutečné hodnoty o jeden krok vpřed/vzad (např. porovnej pred_delta[t] s true_delta[t+1]) a opět vyhodnoť korelaci. Správně fungující model predikuje konkrétní horizont, takže korelace s posunutou (nesprávnou) realizací by měla být výrazně nižší než původní corr_pred_true. Jinými slovy, model by neměl „předpovídat“ i jiný časový posun, což by odhalilo případný leak či nevhodné zarovnání dat.

Strict OOS Test (striktně out-of-sample): Rozděl data na trénovací a testovací část, například 70/30 %. Natrénuj model pouze na prvních 70 % historických dat, zmraz jeho parametry a následně ho otestuj (bez dalšího aktualizování) na zbývajících 30 % úplně neviděných dat. Pokud si model udrží podobný výkon (např. vysokou korelaci) i na takto striktně OOS datech, potvrdí to, že jeho prediktivní schopnost není dána únikem informací ani tím, že by se “učil” během testování. Naopak výrazný pokles výkonu by naznačil, že původní výsledky mohly být nadhodnocené vlivem metody (např. příliš krátkého walk-forward intervalu). Tento test je tedy nejpřísnější potvrzení kauzální čistoty a skutečné prediktivní síly modelu.

Všechny tři testy výše doplňují naši analýzu. Doporučujeme je provést pro definitivní potvrzení, že model nepodvádí pomocí nechtěného leakage a že dosažené metriky odrážejí skutečnou prediktivní schopnost. Pokud model těmito zkouškami projde (shuffle korelace ~0, lag korelace nižší než původní, a solidní výkon na OOS), můžeme s vysokou jistotou potvrdit, že skript je kauzálně čistý a vysoký výkon modelu je reálný.

Závěr: Skript theta_eval_hbatch_jacobi_fixed_leak.py byl navržen správně a podle výše uvedených bodů neobsahuje časový únik dat (leakage) při evaluaci modelu. Model v každém kroku používá pouze historické informace, transformace příznaků a váhy nečerpají z budoucnosti a metriky výkonu jsou počítány až na základě skutečně predikovaných hodnot. Doporučené sanity testy pak mohou dodat dodatečné ujištění, že nejde o artefakt dat či metodologie, ale o validní predikční výkon modelu.