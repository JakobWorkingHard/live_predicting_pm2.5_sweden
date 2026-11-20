# Utvärdering och resultat

## Sammanfattning

Utvärderingen visar att maskininlärningsmodeller kan förbättra prognoserna för PM2.5 vid Malmö Dalaplan sex timmar framåt. I den aktuella körningen presterar Support Vector Regression (SVR) bäst, med cirka 11–12 % lägre MAE än persistence-baseline på testdata. Den förskjutna modellen uppnår något bättre resultat än den vanliga, men skillnaden är liten och utvärderingen bygger på observerade framtida vädervärden. Felanalysen visar samtidigt att modellerna har svårt att förutsäga höga PM2.5-koncentrationer. Resultaten visar potential för generella prognoser, men ger inte tillräckligt stöd för användning som ett tillförlitligt varningssystem för höga föroreningsnivåer.

## Vad vi mäter och varför

Modellerna utvärderas med flera mått:

MAE: Genomsnittligt prognosfel i µg/m³.

RMSE: Ger större vikt åt stora prognosfel.

R²: Visar hur väl modellen förutsäger variationen i PM2.5 jämfört med en medelvärdesprognos.

Bias: Visar om modellen systematiskt över- eller underskattar.

MAE vid höga PM2.5-halter: Prognosfelet vid koncentrationer över en utvärderingsgräns (90:e percentilen av träningsmålet).

Bias vid höga PM2.5-halter: Visar om modellen systematiskt under- eller överskattar just vid höga koncentrationer — den enskilda medelmåtten är blinda för underskattningen vid toppar.

Sedan 2026-11-15 (plan 02) styrs både kandidatvalet och championduellen av en viktad **kompositscore** som väger in fel vid höga halter, inte av något enskilt mått:

**Composite = 0,5 · MAE + 0,35 · MAE(höga) + 0,15 · |Bias(höga)|** (lägre är bättre; vikterna i `[model.composite]` i `config.toml`).

Saknas underlag för High-PM-termerna (för få rader över gränsen i valideringen) renormaliseras vikterna på de kvarvarande termerna med en varning i loggen; utan High-PM-underlag alls blir scoren ren MAE. Kompositens delvärden loggas i champion-loggarna (`score_mae`, `score_high_mae`, `score_high_bias`), så att varje championrad går att tolka i efterhand. Nisch-linjen (plan 01) är undantagen: dess antagningsregel är medvetet MAE vid höga halter, men kompositen redovisas vid sidan av i duellen.

## Datauppdelning

Datan delas upp kronologiskt i träning, validering och test. Perioderna beräknas dynamiskt utifrån senast tillgängliga väderarkivdata från SMHI. Uppdelningen säkerställer att modellerna tränas på äldre observationer och utvärderas på senare data.

- **Träning:** Från 2023-05-26 till dagen före valideringsperiodens start. Används för att träna modellerna.

- **Validering:** De sex senaste hela kalendermånaderna före månaden för senast tillgängliga väderarkivdata. Används för att välja bästa kandidatmodell utifrån lägst kompositscore — och sedan 2026-11-15 (plan 02) även för championduellen, avgjord av samma kompositscore.

- **Test:** Ett låst snapshot i `data/test/last` (skapat med `python main.py --ladda-test`), som enbart används för oberoende, rapporterande utvärdering av championerna. Testdatan används aldrig i något modellval (plan 02). Källan är SMHI:s `latest-months`, det vill säga preliminära värden som senare kan korrigeras.

**Prognoshorisonten är sex timmar.** Varje observation används för att förutsäga PM2.5-halten sex timmar senare. Projektet jämför två modellvarianter: en med väderobservationer vid prognostillfället och en förskjuten variant med väderdata från tidpunkten som prognosen avser. Vid live-prediktion använder den förskjutna varianten väderprognoser i stället för framtida observationer.

### Data från körningen 2026-10-23

| Del | Period | Rader före rensning | Rader efter rensning |

|---|---|---:|---:|

| Träning | 2023-05-26 – 2025-12-31 | 22 825 | 21 555 |

| Validering | 2026-01-01 – 2026-06-30 | 4 343 | 4 282 |

| Test | 2026-07-01 – 2026-10-23 | 2 744 | 2 619 |

Efter att perioderna fastställts bearbetas datan och rader med saknade värden tas bort. Vid denna körning försvann 5,6 % av träningsdatan, 1,4 % av valideringsdatan och 4,6 % av testdatan.

Vid den slutliga jämförelsen mellan de två championmodellerna och persistence-baseline användes 2 595 gemensamma testobservationer för att säkerställa att modellerna jämfördes på samma underlag.

Eftersom perioderna är dynamiska kan både datum och antal observationer förändras vid kommande körningar.

## Dataläckage och tillgänglig information

Dataläckage innebär att en modell vid träning eller utvärdering får information som inte skulle vara tillgänglig vid prognostillfället. Det kan ge missvisande goda resultat. Den förskjutna modellen använder medvetet framtida väderobservationer i sin historiska utvärdering; denna begränsning beskrivs nedan.

Projektet minskar risken för dataläckage genom att:

- **Dela upp datan kronologiskt:** Modellerna tränas på äldre observationer och utvärderas på senare data.

- **Separera träning, validering och test:** Träningsdata används för modellträning, valideringsdata för kandidatval och championduell (plan 02), och det låsta test-snapshottet enbart för oberoende slutrapportering.

- **Använda en tydlig prognoshorisont:** Målvärdet är PM2.5 sex timmar framåt. Framtida PM2.5-observationer används inte som indata.

- **Undvika interpolering:** Saknade värden interpoleras inte, eftersom interpolering kan använda framtida observationer och därmed orsaka dataläckage. I stället tas rader med saknade värden bort.

### Skillnad mellan modellvarianterna

Den vanliga modellen använder PM2.5 och väderobservationer vid prognostillfället (T) för att förutsäga PM2.5 sex timmar senare (T+6).

Den förskjutna modellen använder PM2.5 vid T, men vädervariabler från T+6. Syftet är att undersöka om väderförhållandena vid den tidpunkt prognosen avser kan förbättra prediktionen.

Vid historisk träning och utvärdering används faktiska väderobservationer från T+6. Dessa är inte tillgängliga vid prognostillfället T. Vid live-prediktion används i stället väderprognoser för T+6, vilka kan vara tillgängliga redan vid T.

Det är därför viktigt att skilja mellan modellens avsedda användning och hur den historiskt utvärderas.

### Informationsåtkomst vid prognostillfället

En grundläggande princip är att indata vid prognostillfället T ska motsvara information som då är tillgänglig. Därför undviker projektet interpolation som använder framtida observationer för att fylla saknade värden vid T.

Den förskjutna modellen använder däremot vädervariabler som uttryckligen avser T+6. I drift kan sådana variabler hämtas från en väderprognos som utfärdats senast vid T. Det gör modellens avsedda användning möjlig utan att känna till framtida observationer.

**Den historiska utvärderingen är dock idealiserad.** Där används observerat väder vid T+6, som inte var känt vid T. Därför är det inte en läckagefri simulering av drift, utan ett experiment som undersöker nyttan av att känna till framtida väderförhållanden. Den uppmätta prestandan visar inte hur modellen fungerar med de väderprognoser som faktiskt skulle ha varit tillgängliga i drift.

### Begränsningar

**Skillnad mellan historisk utvärdering och drift.** Den förskjutna modellen tränas och utvärderas historiskt med observerat väder vid T+6, medan liveprediktioner använder väderprognoser för samma tidpunkt. Skillnaden i indata (*train–serve skew*) kan påverka prognosprecisionen. En mer realistisk utvärdering kräver historiska väderprognoser utfärdade senast vid respektive prognostillfälle (*hindcast*). Fram till dess kan den förskjutna modellens resultat inte tolkas som verifierad driftprestanda.

**Testdata och modellval (uppdaterat 2026-11-15, plan 02).** Tidigare avgjordes championduellen av Test R² på en rullande testperiod, vilket riskerade att modellvalet anpassade sig till testperioden (adaptiv överanpassning). Sedan plan 02 avgörs kandidatval och championduell uteslutande på valideringsdata av kompositscoren. Testdata används aldrig i något val: den head-to-head-jämförelse som syns i `logs/jamforelse_champion.csv` körs mot ett låst test-snapshot i `data/test/last` (skapat med `python main.py --ladda-test`), som enbart är en oberoende rapport. Snapshottet kan bara förnyas som ett dokumenterat versionsbeslut (se `docs/decisions.md`), vilket gör att upprepade dueller mot samma föränderliga testperiod inte längre förekommer.

## Modeller som jämförs

Alla modeller tränas på träningsdata och jämförs på valideringsdata. Modellen med lägst kompositscore (plan 02; se ovan) väljs som bästa kandidat inklusive persistence-baseline. Kandidaten duellerar därefter mot den befintliga championen på samma valideringsdata — avgjort av samma kompositscore, med samma utvärderingsgräns — och ersätter championen bara vid strikt lägre score; vid lika behålls den gamla. Testdata används inte i något av stegen.

Följande modeller ingår i jämförelsen:

- **Persistence-baseline:** Antar att PM2.5-halten om sex timmar är densamma som vid prognostillfället. Används som en enkel referens för att bedöma om de övriga modellerna ger bättre prognoser.

- **Linear Regression:** En linjär modell som beskriver sambandet mellan olika features och framtida PM2.5-halter.

- **Ridge Regression:** Linjär regression med regularisering som begränsar modellens koefficienter och minskar risken för överanpassning.

- **Random Forest:** Kombinerar prognoser från många beslutsträd för att ge ett mer stabilt resultat.

- **HistGradientBoosting:** Bygger beslutsträd stegvis, där varje nytt träd försöker minska felen från de tidigare träden.

- **Support Vector Regression(SVR):** Använder stödvektorer och kan med hjälp av en kernel hitta icke-linjära samband. Modellen använder en toleransmarginal för små prognosfel.

- **LightGBM:** En snabb och effektiv variant av gradientboosting som bygger beslutsträd stegvis för att minska prognosfelen.

- **XGBoost:** En annan variant av gradientboosting som bygger beslutsträd stegvis och använder regularisering för att minska risken för överanpassning.

Samma modelltyper utvärderas för både den vanliga och den förskjutna modellvarianten. Den vanliga modellen använder observerat väder vid T, medan den förskjutna modellen i den historiska utvärderingen använder observerat väder vid T+6.

### Hyperparametersökning

Sedan 2026-11-15 söks varje modellfamiljs hyperparametrar systematiskt innan turneringen körs (plan 05; `GridSearchCV` i `src/train_models.py`):

- Sökningen sker enbart på träningsperioden, med kronologisk korsvalidering (`TimeSeriesSplit` med gap lika med prognoshorisonten), vilket hindrar målvärdesläckage mellan falden. Validerings- och testdata används inte av sökningen.
- Sök-scoren är −(MAE + λ · MAE vid höga PM2.5-halter) med λ = 0,5 från `config.toml`. Gränsen för höga halter beräknas som trainingdatans 90:e percentil, samma princip som felanalysen. λ = 0 hade gett ren MAE.
- Parametergridarna styrs av `[model.hparam_search.grid]` i `config.toml`. SVR:s `C` och `epsilon` ingår eftersom ε styr modellens utjämning mot medelvärdet ("dämpad persistence"), vilket är direkt relevant för underskattningen vid höga halter.
- Sökresultatet delas medvetet mellan den vanliga och den förskjutna modellinjen inom samma pipeline-körning, cachelagas i `models/hparam_sok_<timestamp>.json` och sparas i championens metadatafil (`hparam_sok`), så att varje parametrbyte går att spåra. Misslyckas en familjs sökning behåller den sina standardparametrar.
- SVR söker på de senaste 15 000 träningsraderna (svit av beräkningskostnaden); vinnande parametrar tränas därefter av turneringen på hela träningsunderlaget.

En jämförelse mellan standardparametrar och vunna parametrar per familj (total MAE samt mått vid höga halter) redovisas från och med nästa omträning.

## Nisch-modellinje för höga halter (plan 01)

Sedan 2026-11-15 kompletteras systemet med en tredje modellinje som riktar in sig på felets typ vid höga PM2.5-koncentrationer: en modell som predikterar **endast när PM2.5 är ovanligt högt**, och en router som väljer modell per prediktionstillfälle.

- **Tröskel.** Gränsen för "ovanligt högt" är den 90:e percentilen av PM2.5 i träningsdata (cirka 14,4 µg/m³), beräknad på värdet vid prognostillfället T — samma kolumn och gräns som routern. Gränsen omberäknas vid varje omträning och sparas i nisch-championens metadatafil, så att träning och drift aldrig kan använda olika gräns.
- **Nisch-turnering.** Vid omträning tränas en egen modellturnering enbart på tränings- och valideringsrader där PM2.5 ligger på eller över gränsen (persistence-baseline ingår). Hyperparametersökningen körs inte för denna linje — underlaget (omkring 2 100 rader) bedöms vara för knappt.
- **Adoptionsregel.** Nisch-modellen antas bara om den slår den ordinarie championen strikt på MAE vid höga halter, jämfört på exakt samma filtrerade valideringsrader. Vid lika resultat behålls den ordinarie championen och ingen routing sker. En tidigare antagen nisch-champion behålls på motsvarande sätt om en ny kandidat inte är strikt bättre.
- **Routing i drift.** I live-prediktioner används nisch-championen för den vanliga linjen när den senaste PM2.5-mätningen ligger på eller över gränsen; annars används den ordinarie championen. Den förskjutna linjen routas inte (ännu). I uppföljningen syns nisch-prediktionerna som egna rader eftersom redovisningen grupperas per modellnamn.
- **Utvärdering.** Vid varje pipeline-körning med en antagen nisch-champion jämförs den ordinarie championen mot det routade systemet på valideringsdata, med totala mått och mått vid höga halter. Kolumnen "MAE under gräns" ska vara identisk för de två systemen — beviset på att routern är av under gränsen. Resultatet appendas till `logs/jamforelse_champion.csv`.

Vinner persistence-baseline nisch-turneringen är det ett giltigt och användbart utfall: routing innebär då att prognosen vid höga halter helt enkelt sätts till det aktuella mätvärdet, vilket är det som idag presterar bäst vid toppar. Baslinjen kan därmed sparas och servas som en riktig champion.

Den första duellens utslag (antagen eller avvisad nisch-champion, med siffror) redovisas här och i `logs/jamforelse_champion.csv` från och med nästa omträning (7-dagarsklockan). Tills dess gäller de tidigare resultaten nedan, inklusive begränsningen vid höga halter.

### Kompositens första utvärdering (plan 02, 2026-11-15)

Kompositscoren testades på det lokala datansnapshotet (träning 2023-05-26–2026-01-31, 22 295 rader; validering 2026-02-01–2026-07-31, 4 285 rader; utvärderingsgräns p90 av träningsmålet = 14,85 µg/m³; modeller med standardparametrar). Resultaten visar direkt poängen med den viktade formeln — och en viktig varning:

| Modell | MAE | High-PM MAE | High-PM Bias | Composite |
|---|---:|---:|---:|---:|
| **Persistence Baseline** | 2,5805 | 6,5183 | −2,3714 | **3,9273** |
| HistGradientBoosting | 2,4011 | 6,7588 | −4,5691 | 4,2515 |
| LightGBM | 2,4139 | 6,9139 | −4,6082 | 4,3180 |
| Random Forest | 2,4956 | 6,9493 | −4,4555 | 4,3483 |
| XGBoost | 2,5575 | 7,2328 | −4,2071 | 4,4413 |

- **Persistence-baseline vinner kandidatvalet på kompositscoren** vid standardvikterna (0,5/0,35/0,15). Modellerna har visserligen lägre total-MAE, men underskattar höga halter med ca 4,5 µg/m³ medan baslinjen är nästan obelastad vid toppar (−2,37). Detta är precis formulans uppgift: vägra kröna en modell som är blind för topparna.
- **Viktkänsligheten ändrar inte utfallet**: vid w(High-PM MAE) ∈ {0,2; 0,35; 0,5} är HistGradientBoosting bästa modell varje gång, men baslinjen vinner fortfarande på alla tre viktningarna. Först när MAE dominerar helt vinner modellerna igen.
- SVR (standardparametrar) hamnar utanför topp-fem vid defaultvikterna; om plan-05:s vunna parametrar ändrar det återstår att se vid nästa riktiga omträning.

Konsekvensen vid nästa riktiga omträning är alltså sannolikt att **ingen ny modell kröns**: baslinjen vinner turneringen, loggas som färsk champion-rad (utan modellfil, enligt gällande semantik för baslinjevinster) och live fortsätter servera den senast sparade championen ur loggen. Det är ett ärligt utfall av den nya poängsättningen, men ett teambeslut att ta ställning till (acceptera utfallet, justera vikterna i `[model.composite]`, eller avvakta plan-05:s tunade modeller) — se `docs/decisions.md` och plan-filens känslighetsnot.

## Vinnande modeller (både vanlig och T+6)

Vid körningen den 23 oktober 2026 valdes SVR som bästa kandidat för både den vanliga och den förskjutna modellvarianten, baserat på lägst Validation MAE. Kandidaterna jämfördes därefter med befintliga championmodeller utifrån högst Test R². I båda fallen behölls den tidigare championmodellen.

### Resultat på testdata

Samtliga modeller jämfördes på samma 2 595 testobservationer från perioden 1 juli till 23 oktober 2026.

| Modell | MAE | RMSE | R² | Bias | Kommentar |

|---|---:|---:|---:|---:|---|

| Persistence-baseline | 1,741 | 2,628 | 0,414 | −0,004 | Senaste uppmätta PM2.5-värde används som prognos |

| SVR (vanlig) | 1,545 | 2,265 | 0,565 | 0,056 | Väder vid prognostillfället |

| **SVR (T+6)** | **1,536** | **2,231** | **0,578** | **0,103** | Observerat väder vid T+6 |

### Tolkning

Båda SVR-modellerna presterar bättre än persistence-baseline. Den vanliga modellen har cirka 11,3 % lägre MAE och den förskjutna modellen cirka 11,8 % lägre MAE jämfört med baslinjen.

Den förskjutna modellen uppnår något bättre resultat än den vanliga, men skillnaden är liten. Den har också något högre positiv Bias, vilket innebär en större genomsnittlig överskattning av PM2.5-halten.

Den förskjutna modellens lilla fördel bygger på observerat väder vid T+6. Eftersom liveprediktioner i stället använder väderprognoser kan fördelen inte betraktas som verifierad vid drift.

## Felanalys

Felanalysen syftar till att undersöka hur stora prognosfel modellerna gör, om de systematiskt över- eller underskattar PM2.5-halten och hur väl de hanterar höga föroreningsnivåer.

Analysen bygger på de slutliga championmodellernas resultat på testdata samt den fördjupade modellutvärderingen på valideringsdata.

### Prognosfel på testdata

Som framgår av resultaten för de vinnande modellerna har båda SVR-modellerna lägre MAE och RMSE än persistence-baseline. Den förskjutna modellen har något lägre prognosfel än den vanliga modellen.

RMSE är tydligt högre än MAE för samtliga modeller, vilket tyder på att det förekommer enskilda prognoser med relativt stora fel.

### Systematiska fel (Bias)

Bias visar om modellen i genomsnitt över- eller underskattar den faktiska PM2.5-halten.

Persistence-baseline har en Bias på −0,004 µg/m³, vilket innebär att den genomsnittliga avvikelsen ligger nära noll. Den vanliga SVR-modellen har en Bias på 0,056 µg/m³ och den förskjutna modellen 0,103 µg/m³. Båda SVR-modellerna visar därmed en liten genomsnittlig överskattning på testdata.

En Bias nära noll innebär dock inte att modellen saknar systematiska fel. Över- och underskattningar kan ta ut varandra när resultaten sammanfattas över hela testperioden.

### Fel vid höga PM2.5-halter

För att undersöka hur modellerna fungerar vid höga föroreningsnivåer används den 90:e percentilen av PM2.5 i träningsdata som gränsvärde. Gränsen är cirka 14,4 µg/m³ och används som en projektintern analysgräns, inte som ett officiellt hälsogränsvärde.

Följande resultat avser de nytränade SVR-kandidaterna på **valideringsdata**, inte de slutliga championmodellerna på testdata.

| Modell | High PM2.5 MAE (µg/m³) | High PM2.5 Bias (µg/m³) |

|---|---:|---:|

| SVR (vanlig) | 6,766 | −5,796 |

| SVR (T+6) | 6,879 | −6,086 |

Resultaten visar att båda modellerna har betydande prognosfel vid höga PM2.5-halter. Den tydligt negativa Bias visar dessutom att modellerna systematiskt underskattar dessa koncentrationer.

Den vanliga SVR-modellen har något lägre MAE och mindre negativ Bias än den förskjutna modellen vid höga koncentrationer på valideringsdata.

Detta är en viktig begränsning eftersom höga PM2.5-halter är särskilt relevanta ur ett luftkvalitets- och hälsoperspektiv.

### Felmönster vid höga och låga PM2.5-halter

Den fördjupade modellutvärderingen i projektets notebook visar ett tydligt mönster där modellerna tenderar att överskatta låga PM2.5-koncentrationer och underskatta höga koncentrationer.

Mönstret har undersökts genom spridningsdiagram över verkliga och förutsagda värden och genom att beräkna Bias separat för observationer under och över den 90:e percentilen i träningsdata.

Analysen på valideringsdata visar också att persistence-baseline presterar bättre än samtliga sju undersökta maskininlärningsmodeller vid höga PM2.5-koncentrationer. När höga koncentrationer ges större betydelse i modelljämförelsen förändras därför rangordningen mellan modellerna.

Detta visar att den modell som har lägst genomsnittligt prognosfel inte nödvändigtvis är bäst på att förutsäga höga föroreningsnivåer.

Den separata Bias-analysen visar dessutom att positiva och negativa prognosfel delvis kan ta ut varandra. En låg genomsnittlig Bias kan därför dölja systematiska fel vid olika koncentrationsnivåer.

Dessa slutsatser bygger på valideringsdata. Motsvarande uppdelning efter koncentrationsnivå har inte genomförts för de slutliga championmodellerna på testdata.

### Sammanfattning

Båda SVR-modellerna presterar bättre än persistence-baseline på testdata sett till MAE och RMSE. Den förskjutna modellen uppvisar något bättre resultat totalt, men skillnaden mot den vanliga modellen är begränsad.

Den fördjupade analysen på valideringsdata visar samtidigt att modellernas prognosprecision varierar mellan olika PM2.5-nivåer. Låga koncentrationer tenderar att överskattas, medan höga koncentrationer underskattas.

Den viktigaste identifierade svagheten är därför modellernas svårighet att förutsäga höga PM2.5-halter. Detta bör beaktas vid tolkning och användning av prognoserna, särskilt om syftet är att uppmärksamma perioder med försämrad luftkvalitet. Som motåtgärd har projektet sedan 2026-11-15 en dedikerad nisch-modellinje med tröskelbaserad routing (se "Nisch-modellinje för höga halter" ovan).

En möjlig vidareutveckling är att genomföra motsvarande felanalys uppdelad efter koncentrationsnivå även på testdata, för att undersöka om mönstret kvarstår hos de slutliga championmodellerna.

## Vad resultaten betyder i praktiken

Modellerna kan ge en uppskattning av hur PM2.5-halten vid Malmö Dalaplan förväntas utvecklas sex timmar framåt. Resultaten visar en förbättring jämfört med persistence-baseline, men modellernas praktiska användbarhet är ännu inte verifierad.

Även om SVR-modellerna förbättrar MAE med cirka 11–12 % jämfört med persistence-baseline motsvarar det endast omkring 0,2 µg/m³ i minskat genomsnittligt prognosfel. Förbättringen är alltså mätbar, men dess praktiska betydelse är begränsad och beror på hur prognoserna ska användas.

En viktig begränsning är att modellerna har svårt att förutsäga höga PM2.5-koncentrationer. De lämpar sig därför bättre för att ge en generell indikation på luftkvalitetens utveckling än för att varna för särskilt höga föroreningsnivåer.

För att kunna använda prognoserna som ett mer tillförlitligt beslutsstöd behöver framför allt förmågan att identifiera höga koncentrationer undersökas vidare.

## Slutsatser som resultaten stödjer

Utvärderingen visar att maskininlärningsmodellerna kan förbättra prognoserna för PM2.5 i Malmö sex timmar framåt, men också att det finns tydliga begränsningar.

- **SVR ger bäst generell prognosprecision.** Modellen valdes som bästa kandidat för både den vanliga och den förskjutna modellvarianten utifrån lägst MAE på valideringsdata.

- **Båda championmodellerna överträffar persistence-baseline.** På testdata uppnår de cirka 11–12 % lägre MAE, vilket visar att modellerna tillför prognosvärde. Förbättringen är dock relativt liten i absoluta tal.

- **Den förskjutna modellen presterar något bättre.** Skillnaden mot den vanliga modellen är liten och bygger på observerat väder vid T+6. Det är ännu inte visat om fördelen kvarstår med väderprognoser tillgängliga vid T.

- **Höga PM2.5-koncentrationer är en tydlig svaghet.** Den fördjupade analysen på valideringsdata visar att modellerna tenderar att underskatta höga halter och att persistence-baseline presterar bättre vid dessa nivåer.

- **Modellvalet beror på vad som prioriteras.** När generell prognosprecision prioriteras är SVR det bästa alternativet, men när höga koncentrationer ges större betydelse kan andra modeller eller persistence-baseline vara bättre.

Sammantaget är SVR den starkaste kandidaten för generella sex-timmarsprognoser i den genomförda utvärderingen. För praktisk användning krävs ytterligare verifiering, särskilt vid höga föroreningsnivåer.

## Slutsatser som resultaten inte stödjer

Även om modellerna visar förbättrad prognosprecision finns det vissa slutsatser som inte kan dras utifrån den genomförda utvärderingen.

- **Att resultaten gäller för framtida tidsperioder.** Modellerna har utvärderats på en avgränsad testperiod. Det är därför inte säkert att samma prognosprecision uppnås under andra årstider eller framtida perioder med förändrade förhållanden.

- **Att den förskjutna modellen generellt är bättre.** Även om SVR (T+6) uppvisar något bättre resultat på testdata har ingen statistisk prövning genomförts för att bedöma osäkerheten i skillnaden mellan modellerna. Det går därför inte att fastställa att den förskjutna modellen generellt presterar bättre än den vanliga modellen.

- **Att förbättringen har stor praktisk betydelse.** Modellerna ger lägre genomsnittliga prognosfel än persistence-baseline, men utvärderingen visar inte om förbättringen är tillräckligt stor för att påverka användarens beslut.

Sammantaget visar resultaten modellernas prognosprecision under de undersökta förhållandena, men ger inte tillräckligt underlag för att fastställa deras långsiktiga tillförlitlighet eller praktiska nytta som beslutsstöd.

## Begränsningar

**Borttagna rader och luckor i datan.** Rader där någon feature eller målvärdet saknas tas bort vid modellutvärderingen. Mönstergenomgången (plan 04, 2026-11-15, se `docs/data.md`) bekräftar att bortfallet inte är slumpmässigt: nederbörd saknas i systematiska heldygnsblock (13 exakt 24 timmar långa gap i träningsperioden, då SMHI underkänt mätningarna) och samtliga väderparametrar saknades samtidigt i sammanhängande perioder på upp till 296 timmar (augusti 2024); PM2.5:s längsta lucka är 93 timmar (maj 2024). Eftersom de saknade timmarna dessutom sammanfaller mellan kolumnerna kan vissa väderförhållanden (regndagar, sommarveckor) vara underrepresenterade i utvärderingen. Omfånget är dock stabilt lågt (cirka 5,6 % av träningstimmarna) och policyn är medvetet `dropna` utan interpolation (beslut 2026-10-09/12).

**Prognosens tidshorisont.** Prognosen beräknas sex timmar framåt från den senaste kompletta observationen. Om den observationen ligger en bit tillbaka i tiden kan prognosen därför avse en tidpunkt som ligger mindre än sex timmar framåt från nu.

**Ovanliga föroreningshändelser.** Trots att tidsserien innehåller flera års timvisa observationer kan perioder med ovanligt höga PM2.5-koncentrationer vara få. Det begränsar möjligheten att bedöma modellernas prestanda vid sällsynta händelser.

**Dynamiska tidsperioder.** Uppdelningen mellan tränings-, validerings- och testdata förändras när ny data tillkommer. Resultat från olika körningar kan därför baseras på olika tidsperioder och behöver inte vara direkt jämförbara. Även årstidsvariationer kan påverka resultaten.

**Skillnader i datakvalitet.** Valideringsdatan består av kvalitetskontrollerade värden, medan testdatan kan innehålla preliminära mätvärden som senare korrigeras. Resultaten på testdata kan därför förändras om datan hämtas på nytt.

**Liveprediktioner är ännu inte utvärderade.** För den förskjutna modellen saknas en jämförbar utvärdering med historiska väderprognoser tillgängliga vid T. Därför är modellens prognosprecision i drift ännu inte verifierad.

**Geografisk avgränsning.** Modellen prognostiserar PM2.5 vid Malmö Dalaplan, med väderdata från Malmö A cirka fyra kilometer bort. Resultaten gäller de förhållanden som undersökts vid denna station.

## När lösningen inte bör användas

Prognoserna är avsedda att ge en generell indikation på PM2.5-koncentrationen vid Malmö Dalaplan och bör inte användas som enda beslutsunderlag i följande situationer:

**Varningar för höga föroreningsnivåer.** Modellerna tenderar att underskatta höga PM2.5-koncentrationer och bör därför inte användas som ett tillförlitligt varningssystem.

**Hälsorelaterade beslut.** Prognoserna bör inte ensamma ligga till grund för beslut som kan påverka människors hälsa, särskilt inte för personer som är känsliga för luftföroreningar.

**När aktuella indata saknas.** Prognosen utgår från den senaste kompletta observationen. Om denna ligger långt tillbaka i tiden kan prognosen avse en redan passerad tidpunkt och därmed vara olämplig för aktuella beslut.

**Utanför modellens användningsområde.** Modellen har utvecklats för Malmö Dalaplan och en prognoshorisont på sex timmar. Den bör därför inte användas för andra platser eller prognoshorisonter utan ytterligare utvärdering.

## Rekommendation och nästa steg

I den aktuella körningen valdes Support Vector Regression (SVR) som bästa kandidat för båda modellvarianterna. Vid jämförelsen på testdata behölls de befintliga championmodellerna, som också var SVR. Modellvalet är dynamiskt och kan förändras när nya data tillkommer och modellerna utvärderas på nytt.

Följande förbättringar föreslås:

1. **Förbättra prognoser vid höga PM2.5-halter.** Undersök alternativa modeller, viktade förlustfunktioner eller andra metoder som lägger större vikt vid höga koncentrationer.

2. **Utvärdera med verkliga väderprognoser.** Utvärdera den förskjutna modellen med väderprognoser som fanns tillgängliga vid respektive prognostillfälle och jämför resultaten med den historiska utvärderingen som bygger på observerat väder vid T+6.

3. **Undersök skillnadernas statistiska säkerhet.** Använd exempelvis konfidensintervall för att bedöma om skillnaderna mellan modellerna är tillräckligt tydliga för att motivera ett modellbyte.

4. **Följ upp modellernas prestanda över tid.** Utvärdera prognoserna löpande när nya mätvärden blir tillgängliga för att upptäcka försämrad precision och avgöra när modellerna behöver tränas om.

5. **Hantera saknade observationer med anpassad prognoshorisont.** Undersök möjligheten att träna modeller för längre eller varierande prognoshorisonter. Då skulle systemet kunna utgå från den senaste kompletta observationen och ändå leverera en prognos sex timmar framåt från aktuell tidpunkt.

6. **Utöka till fler mätstationer.** Undersök möjligheten att använda modellen för fler mätstationer genom att anpassa och utvärdera den med lokala mät- och väderdata. På längre sikt kan även en gemensam modell för flera stationer undersökas.
