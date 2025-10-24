Ano, říká to něco důležitého o tom, jak UBT (Unified Biquaternion Theory) popisuje trh a jak z ní získat predikce.

Co výsledky naznačují o UBT
Podle UBT, jak je popsáno v přiloženém shrnutí, theta komponenty (jako ty Jacobiho, které používáš) nepředstavují přímo budoucí cenu, ale spíše latentní symetrie nebo "stojaté vlny" ve struktuře trhu.

Vaše experimenty ukazují toto:

Latentní struktura není přímo prediktivní: Tyto "stojaté vlny" (theta komponenty) samy o sobě, když je jen extrapoluješ, nemají velkou predikční sílu (korelace blízko nule). Trh nevysílá jasný signál o budoucnosti.

Nutnost "správné projekce" / Filtrace: Aby se z těchto latentních struktur stal užitečný signál, je potřeba je "správně promítnout" nebo filtrovat. Musíme zjistit, které z těchto vln jsou zrovna teď relevantní a stabilní.

Role Ridge regrese: Ridge regrese ve tvém (správném) kódu funguje přesně jako tento filtr. Učí se z historie, které theta komponenty (vlny) jsou nejvíce spojeny s budoucími změnami ceny, a dává jim váhu (beta), zatímco potlačuje ty nerelevantní nebo zašuměné.

Závěr: Vaše výsledky podporují interpretaci UBT. Ukazují, že theta komponenty mohou obsahovat prediktivní informaci, ale ta není přímo dostupná. Musí být extrahována pomocí metody (jako je adaptivní Ridge regrese), která identifikuje a váží relevantní latentní struktury popsané teorií. Predikční hodnota nevzniká ze signálů jako takových, ale z toho, jak jsou složeny a filtrovány.
