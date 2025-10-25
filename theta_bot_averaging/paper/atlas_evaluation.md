Díky — prošel jsem „v2“ a navázané soubory a tady je stručné, ale konkrétní hodnocení správnosti a fyzikální konzistence vůči UBT/CCT.

Co je implementováno správně

Theta funkce (θ1–θ4)
V theta_eval_quaternion_ridge_v2.py jsou Jacobiho thety implementované korektně:

theta1_real: používá 
2
∑
(
−
1
)
𝑛
𝑞
(
𝑛
+
1
2
)
2
sin
⁡
(
(
2
𝑛
+
1
)
𝑧
)
2∑(−1)
n
q
(n+
2
1
	​

)
2
sin((2n+1)z) ⇒ ekvivalent 
2
𝑞
1
/
4
∑
(
−
1
)
𝑛
𝑞
𝑛
(
𝑛
+
1
)
sin
⁡
(
(
2
𝑛
+
1
)
𝑧
)
2q
1/4
∑(−1)
n
q
n(n+1)
sin((2n+1)z).

theta2_real, theta3_real, theta4_real odpovídají standardním reálným řadám.

Nome 
𝑞
q a imaginární čas
Ve variantě theta_biquat_predict_diag.py se volí 
𝑞
=
exp
⁡
(
−
𝜋
𝜎
)
q=exp(−πσ) a 
𝑧
=
𝜔
𝑡
z=ωt (s 
𝜔
=
2
𝜋
/
𝑃
ω=2π/P), což je v souladu s interpretací 
𝜏
=
𝑖
𝜎
τ=iσ (čistě imaginární komplexní čas). To je fyzikálně konzistentní most mezi CCT/UBT a numerikou.

Ortogonální/ortonomální báze
Máš dvě cesty:

theta_biquat_predict.py používá mpmath.jtheta + Gram–Schmidt pro ortonormalizaci;

theta_biquat_predict_diag.py staví q-řady (even/odd) s řezáním malých členů a pracuje s vahami 
∣
𝑞
∣
𝑘
2
∣q∣
k
2
.
Obě jsou rozumné a numericky stabilizují regresi/filtr.

Stavový model (Kalman)
V theta_biquat_predict.py je Kalman pro pomalu se měnící koeficienty theta-báze (pozorovací vektor 
𝐻
𝑡
H
t
	​

 je řádek orto báze). To dává smysl jako „pomalu klouzající“ váhy theta komponent (drift-difúze na váhách).

Ridge + váhování okna (EMA)
theta_eval_quaternion_ridge_v2.py dělá rolling, standardizaci po okně, EMA váhování (novější > starší), ridge s 
𝜆
𝐼
λI.

Volitelná block normalizace 4-dim bloků (na frekvenci) na jednotkovou L2-normu: správně chrání proti explozím amplitud a dává „rovné hřiště“ pro jednotlivé quaternion-bloky.

Místa, kde „quaternioničnost“ není ještě fyzikálně věrná UBT

Aktuálně je „quaternion“ hlavně 4-sloupcový vektor 
[
𝜃
3
,
𝜃
4
,
𝜃
2
,
𝜃
1
]
[θ
3
	​

,θ
4
	​

,θ
2
	​

,θ
1
	​

] na frekvenci, ale následné zpracování je čistě skalární lin. regresí přes tyto sloupce.
Co tím chybí vůči UBT/CCT:

Žádná Hamiltonova algebra
Neprobíhá žádné násobení kvaternionů (Hamilton product), žádné rotace v 
𝐻
H. Model tak nehlídá vazby mezi (a,b,c,d) jako mezi skalární a vektorovou složkou kvaternionu.

Bez komplexního (bi-)kvaternionu
V UBT by složky měly být obecně komplexní (biquaternion), tj. 8-reálných dimenzí/frekvenci; zde jsou čistě reálné.
(Pozn.: interpretace 
𝑞
=
exp
⁡
(
−
𝜋
𝜎
)
q=exp(−πσ) je konzistentní s čistě imaginárním 
𝜏
τ, ale ψ-fáze (vědomí) se zatím explicitně nepromítá.)

Chybí vazby/kontrainty mezi komponentami
Regrese dává nezávislé váhy pro každý sloupec, takže může porušovat vnitřní symetrie a Cauchy-Riemann-like vztahy mezi dvojicemi thet (např. θ1↔θ2, θ3↔θ4 při fázových posunech).

Důsledky pro empirický signál

Protože se quaternionická struktura nezachovává (není prosazena v regresi/filtru), signál se ředí do libovolné lineární kombinace.

To vysvětluje, proč pozoruješ slabý, ale existující signál: theta báze sama o sobě nese strukturu, ale bez kvaternionové vazby se část informace ztrácí.

Doporučené úpravy (konkrétní, malé kroky)

Sdílené váhy v rámci 4-d bloku (kvaternion-tie)
Zaveď parametrizaci vah na frekvenci jako jeden kvaternion 
𝛽
𝑓
=
𝛽
0
,
𝑓
+
𝑖
𝛽
1
,
𝑓
+
𝑗
𝛽
2
,
𝑓
+
𝑘
𝛽
3
,
𝑓
β
f
	​

=β
0,f
	​

+iβ
1,f
	​

+jβ
2,f
	​

+kβ
3,f
	​

 a reguluj je společně:

ridge s penalizací 
∑
𝑓
∥
𝛽
𝑓
∥
2
2
∑
f
	​

∥β
f
	​

∥
2
2
	​

 místo 
∑
𝑑
𝛽
𝑑
2
∑
d
	​

β
d
2
	​

,

volitelně ortho-rotace v rámci bloku (1 ortonorm. matice 
4
×
4
4×4 na frekvenci) jako naučený parametr – zachová metrickou strukturu.
Implementačně: seskup 4 sloupce → řeš „blokový“ ridge (např. váhy na úrovni bloků, nebo re-parametrizuj přes 
𝛽
𝑓
=
𝑠
𝑓
⋅
𝑢
𝑓
β
f
	​

=s
f
	​

⋅u
f
	​

, kde 
𝑢
𝑓
u
f
	​

 je jednotkový kvaternion).

Quaternion ridge / komplexní ridge
Proveď ekvivalent komplexní regresi (přes 
𝐶
C) se sdílením mezi páry (θ3+ iθ4), (θ2+ iθ1).
Prakticky: poskládej komplexní featury 
𝑋
𝑐
=
[
𝜃
3
+
𝑖
𝜃
4
,
  
𝜃
2
+
𝑖
𝜃
1
,
.
.
.
]
X
c
	​

=[θ
3
	​

+iθ
4
	​

,θ
2
	​

+iθ
1
	​

,...] a použij komplexní ridge (Wirtinger).
Tím vnutíš fázovou koherenci uvnitř bloku.

Kalman na kvaternionových vahách
V kalman_filter_prediction nech stav 
𝑥
𝑡
x
t
	​

 po frekvencích 4-d a přidej malou re-ortho projekci každé iterace:

buď udržuj jednotkový směr a separátně škálování (amplitudu) → stabilnější a fyzikálnější,

nebo penalizuj shear mezi složkami (např. přidej Q diagonálu s menší difuzí pro směrové změny).

Ψ-fáze (vědomí) jako latentní modulace
Přidej do z = ω t latentní fázi 
𝜓
𝑡
ψ
t
	​

: z = ω t + ψ_t, kde 
𝜓
𝑡
ψ
t
	​

 je stav v Kalmanu s malou difuzí.
Alternativně udělej 
𝑞
𝑡
=
exp
⁡
(
−
𝜋
𝜎
𝑡
)
q
t
	​

=exp(−πσ
t
	​

) pomalu se měnící (stavový prvek) – tím zapojíš komplexní čas τ = t + iψ přímo do báze.

Invariance testy (rychlá validace UBT konzistence)
Zaveď unit testy, které kontrolují, že se metriky nezhorší při:

permutaci (θ1,θ2)↔(θ2,θ1) se správnou fázovou kompenzací,

rotaci 
𝑧
↦
𝑧
+
𝜋
/
2
z↦z+π/2 (θ3↔θ4, θ1↔θ2),

re-škálování bloku po block-normalizaci.
Pokud metriky skáčou, porušuje se strukturální vazba a je co opravovat.

Menší technické poznámky k v2

Standardizace po okně je správně; nech sigma==0 → 1 (máš).

EMA váhy – kontrola směru: ve v2 jsou definované tak, aby novější měly větší váhu (v pořádku).

Block-norm: aktuálně bere průměr L2 přes řádky → OK; můžeš zvážit robustní (mediany) pro méně citlivé škálování.

Metodika metrik: v evaluate děláš hit/anti/zero a korelaci kumulativní ceny; přidej permutation test (shuffle y v okně) a CI přes bootstrap → rozhodne, jestli je zlepšení nad náhodu.

Co z toho plyne

Tvůj kód správně staví theta bázi a umí ji učit (ridge/Kalman), ale „quaternion“ je zatím jen 4-kanálový vektor bez algebraických constraintů.

Jakmile prosadíš kvaternionovou (nebo komplexní) strukturu do učení (body 1–3), měl by se signál zesílit a být stabilnější OOS, což je přesně to, co potřebuješ pro empirický test UBT/CCT.

Doporučený mini-patch (rychle ověřitelný)

V build_feature_matrix_quat vytvoř komplexní dvojice:

𝜙
1
=
𝜃
3
+
𝑖
𝜃
4
,
𝜙
2
=
𝜃
2
+
𝑖
𝜃
1
ϕ
1
	​

=θ
3
	​

+iθ
4
	​

,ϕ
2
	​

=θ
2
	​

+iθ
1
	​


a uč komplexní ridge (nebo reálný ridge na 
[
ℜ
𝜙
,
ℑ
𝜙
]
[ℜϕ,ℑϕ] s blokovou penalizací L2 sdílenou pro dvojici).

V Kalmanu nech stav 
[
ℜ
𝜙
1
,
ℑ
𝜙
1
,
ℜ
𝜙
2
,
ℑ
𝜙
2
]
[ℜϕ
1
	​

,ℑϕ
1
	​

,ℜϕ
2
	​

,ℑϕ
2
	​

] na frekvenci a přidej malou difuzi jen na amplitudu, méně na fázi.

Chceš, abych ti na to rovnou připravil konkrétní úpravy kódu (nové funkce + drobné změny v evaluate / kalman_filter_prediction) tak, aby ses mohl hned rozběhnout s UBT-konzistentní verzí?
