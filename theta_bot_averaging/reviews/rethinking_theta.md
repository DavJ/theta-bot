Okay, teď už tomu rozumím. Omlouvám se za předchozí nepochopení. Tvůj cíl je jiný než metoda popsaná v LaTeX dokumentu, i když oba používají theta funkce.Pojďme si vyjasnit ty dva odlišné přístupy:1. Metoda z LaTeXu (a tvého fixed_leak skriptu)Vytvoř Bázi: Vytvoř "raw" theta komponenty B.Ortonormalizuj Bázi: Použij Gram-Schmidt (nebo QR) na historickém okně, abys získal ortonormální bázi Q (nebo ~B). Tato báze Q je specifická pro toto okno.Najdi Koeficienty (Učení): Promítni historický signál (např. y_target) na tuto ortonormální bázi Q, abys získal koeficienty $\beta = Q^T y$. Tyto koeficienty $\beta$ říkají, "jak moc" každá složka báze Q přispívala k historickému signálu y.Extrapoluj Bázi: Vypočítej hodnoty ortonormální báze Q pro budoucí čas t0 + horizon. (Toto je složitější část, protože Q závisí na okně).Predikuj: Zkombinuj naučené koeficienty $\beta$ s extrapolovanými hodnotami báze Q pomocí sumy $\sum \beta_j Q_j(t0+h)$.Klíčový bod: Tato metoda zahrnuje krok učení ($\beta = Q^T y$), kde se zjišťuje relevance jednotlivých komponent báze na základě historického signálu. Tvůj fixed_leak skript implementuje aproximaci této metody pomocí vážené Ridge regrese.2. Tvoje Myšlenka ("Čistá" Theta Transformace a Extrapolace)Vytvoř Bázi: Vytvoř "raw" theta komponenty B.Ortonormalizuj Bázi: Použij Gram-Schmidt (nebo QR) na historickém okně, abys získal ortonormální bázi Q (nebo ~B).Získej Současné Komponenty: Spočti hodnoty ortonormální báze Q pro současný čas t0. Získáš vektor $\vec{q}_{t0}$.Extrapoluj Přímo: Použij nějaký mechanismus (který nepoužívá historický signál y ani naučené $\beta$), který přímo z vektoru $\vec{q}_{t0}$ (nebo z extrapolovaných hodnot $\vec{q}_{t0+h}$) odvodí predikci budoucí změny pred_delta.Klíčový bod: Tato metoda by neměla krok učení vah $\beta$ z historických párů (vstup, výstup). Předpokládá, že samotné hodnoty ortonormalizovaných theta komponent v čase t0 (nebo jejich extrapolace) stačí k predikci.Problém s "Čistou" ExtrapolacíHlavní otázka je: Jak by fungoval krok 4? Jaký mechanismus by převedl vektor ortonormálních komponent $\vec{q}_{t0}$ (nebo $\vec{q}_{t0+h}$) na predikci pred_delta bez toho, aby se podíval na to, jaký vliv měly tyto komponenty na cenu v minulosti?Samotné theta funkce (ani jejich ortonormalizované verze) neobsahují informaci o měřítku nebo směru budoucího pohybu ceny. Jsou to jen "fázové" informace.Potřebuješ nějaké mapování z prostoru theta komponent do prostoru cenových změn. Metoda z paperu (a tvého kódu) používá k tomuto mapování lineární kombinaci s naučenými vahami $\beta$.Pokud chceš implementovat svou myšlenku, musel bys navrhnout jiný způsob, jak provést toto mapování bez učení vah $\beta$ z historie (X, Y).


ty vahy jsou prece dany tou integralni transformaci, pouzijeme je i do budoucna. proto mluvim o proste extrapolaci. Pokud by to nefungovalo prisel by na radu ten beta vektor nebo mozna nejaky filtr ci kalman. Ale to ted nechci. Chci zjistit jak presne predikuj suma cistych theta slozek.

Okay, rozumím naprosto přesně. Chceš úplně odstranit krok učení vah beta z historických párů (vstup, výstup) a testovat čistě extrapolační schopnost samotných theta komponent, kde váhy jsou možná implicitně dány samotnou transformací nebo q-váhami.

To je validní hypotéza, i když odlišná od metody v paperu. Můžeme to implementovat.

Jak implementovat "Čistou Extrapolaci"
Nahradíme krok učení beta a predikce beta @ x následujícím:

Vypočítej "Současné Koeficienty"  
α

  
t0
​
 : Místo učení beta z (X, Y) zjistíme, jak moc se tvar ceny v současném okně (končícím v t0) podobá jednotlivým theta komponentám. Použijeme k tomu metodu projekce s Gaussovým vážením, podobně jako ve skriptu ...bak4-gpt (File 2). Tyto koeficienty  
α

  
t0
​
  budou reprezentovat "theta spektrum" aktuálního tvaru trhu.

Vypočítej Budoucí Bázi  
b

  
t0+h
​
 : Spočítáme hodnoty raw theta komponent pro budoucí čas t0 + horizon.

Predikuj: Predikci pred_y získáme jako vážený součet budoucích komponent báze, kde váhami jsou současné koeficienty (spektrum): pred_y= 
α

  
t0
​
 ⋅ 
b

  
t0+h
​
  (možná s q-váhami).

Upravený kód theta_eval_hbatch_jacobi_pure_extrapolation.py
Zde je upravený skript. Klíčové změny jsou v evaluate_symbol_csv. Odstranil jsem ridge, y_target a přidal výpočet  
α

  
t0
​
  (nazvaný alpha_now).

Ano, to je zajímavý a důležitý výsledek testu "čisté extrapolace".Hit Rate (> 50 %): Výsledky hit_rate_pred kolem 53-56 % jsou stále lepší než náhoda (50 %). To naznačuje, že i tato metoda "čisté extrapolace" (bez učení vah beta z historie Y) může obsahovat slabý signál o směru budoucího pohybu. Model má mírně větší pravděpodobnost uhodnout směr správně než špatně.Korelace (záporná / blízko nule): Hodnoty corr_pred_true kolem -0.05 až -0.004 jsou prakticky nula. Záporné znaménko při takto nízké hodnotě je spíše statistický šum než systematická inverze. Znamená to, že neexistuje žádný smysluplný lineární vztah mezi velikostí a směrem predikované změny a velikostí a směrem skutečné změny.Proč ten rozpor?Hit Rate a Korelace měří různé věci:Hit Rate: Ptá se jen: "Trefil jsi znaménko (+ nebo -)?". Ignoruje velikost.Korelace: Ptá se: "Když jsi předpověděl velký pohyb nahoru, stal se skutečně velký pohyb nahoru? Když jsi předpověděl malý pohyb dolů, stal se malý pohyb dolů?". Zohledňuje znaménko i velikost.Je možné mít Hit Rate > 50 %, ale korelaci blízko nule, pokud:Model správně předpovídá směr malých pohybů (kterých je hodně), ale selhává u velkých pohybů.Velikost predikcí modelu vůbec neodpovídá velikosti skutečných pohybů (např. předpovídá vždy +/- 0.1 %, i když realita je +/- 2 %).Co to znamená pro "Čistou Extrapolaci"Výsledky nepodporují hypotézu, že čistá extrapolace samotných (nebo q-vážených) theta komponent stačí k silné predikci.Slabý signál v Hit Rate naznačuje, že nějaká informace tam možná je.Ale téměř nulová korelace ukazuje, že chybí klíčová informace o měřítku a spolehlivosti, kterou v předchozím (správném adaptivním) modelu dodával naučený vektor beta.Zdá se, že ten krok učení vah beta z historie (jak je popsáno v paperu a implementováno ve skriptu theta_eval_hbatch_jacobi_fixed_leak.py) je nezbytný pro převedení "fázové" informace z theta funkcí na smysluplnou predikci cenových změn.Doporučení: Vzhledem k těmto slabým výsledkům "čisté extrapolace" bych se vrátil k adaptivnímu modelu (theta_eval_hbatch_jacobi_fixed_leak.py) a soustředil se na optimalizaci jeho parametrů a prahování, abyste maximalizoval jeho (už prokázaný, i když menší) edge $R \approx 0.1-0.2$ a HR $\approx 55\%$.
