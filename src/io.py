import pandas as pd
import joblib
import json
from pathlib import Path
from datetime import datetime, timedelta
from src.logger import setup_logger

log = setup_logger(__name__)


def generate_csv(data, path_and_name_of_csv, index = True):
    """Försöker spara en DataFrame till en CSV-fil. 
    Om filen redan existerar, görs ingenting.
    """
    try:
        # mode='x' gör att pandas kastar ett FileExistsError om filen redan finns
        data.to_csv(path_and_name_of_csv, mode='x', index=index)
        print(f"Skapade: {path_and_name_of_csv,}")
        
    except FileExistsError:
        # Filen finns redan, vi avbryter tyst
        pass

def merge_df_and_generate_csv(data1_path, data2_path, on_column: str, path_and_name_of_csv, index, how = "outer"):
    """ Tar emot två df filer och mergar ihop dessa baserat på
    kolumn: on_column
    genererar sedan en mergad csv
    NOTERING: Fungerar endast om kolumnamnen är detsamma i båda dfs:en"""
    if path_and_name_of_csv.is_file():
        return None

    df1 = pd.read_csv(data1_path)
    df2 = pd.read_csv(data2_path)
    merged_df = pd.merge(df1, df2, on=on_column, how=how)
    return generate_csv(merged_df, path_and_name_of_csv, index)


def save_champion_model(
    model,
    model_name,
    best_row,
    models_dir,
    random_state,
    metric,
    filprefix="",
    bast_params=None,
    threshold=None,
    threshold_rule=None,
    vikter=None,
    utvarderings_grans=None,
):
    """
    Sparar den vinnande modellen och en JSON-fil med metadata.

    Parametrar
    ----------
    model : estimator
        Den tränade och fit:ade modellen.
    model_name : str
        Namn på modellen, t.ex. "Linear Regression".
    best_row : pd.Series
        Raden för vinnaren från results_df (innehåller alla metriker).
    models_dir : Path
        Mapp där modellen ska sparas. Skapas om den inte finns.
    random_state : int
        Frö som användes vid träning.
    metric : str
        Metrikk som avgjorde valet, t.ex. "Validation Composite".
    filprefix : str, valfri
        Prefix på filnamnen, t.ex. "forskjuten_" för den förskjuna
        modellinjen. Gör det möjligt att skilja varianterna åt i
        models-mappen och vid inläsning av champion-modellen.
    bast_params : dict, valfri
        Vunna hyperparametrar från hparam-sökningen (plan 05) för den
        här modellfamiljen, t.ex. {"C": 4.0, "epsilon": 0.1}.
        None/{} => standardparametrar användes.
    threshold : float, valfri
        Gränsen för höga PM2.5-värden (plan 01). Skickas endast av
        nisch-linjen; sparas i metadata så att routern i live och
        utvärderingen i träning GARANTERAT använder samma gräns.
        None => ingen threshold-nyckel skrivs (ordinarie linjer).
    threshold_rule : str, valfri
        Namn på regeln som producerade threshold, t.ex.
        "pm25_p90_training" ([model.high_pm] i config.toml).
    vikter : dict, valfri
        Kompositvikterna (plan 02, [model.composite] i config.toml)
        som scoren i champion-loggen/metadata bygger på.
    utvarderings_grans : float, valfri
        Utvärderingsgränsen för High-PM-måtten (plan 02: p90 av
        träningsmålet y_train). Skilj den från nisch-linjens
        threshold (routningsgränsen) ovan — två olika gränser.
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y_%m_%d_%H-%M")
    safe_name = model_name.replace(" ", "_").lower()
    metris_namn = metric.replace(' ','_').replace("²", "2")
    model_path = models_dir / f"{filprefix}{safe_name}_{metris_namn}_{timestamp}.joblib"
    meta_path = models_dir / f"{filprefix}{safe_name}_{metris_namn}_{timestamp}.json"

    joblib.dump(model, model_path)
    print(f"Sparade modell: {model_path}")

    meta = {
        "model_name": model_name,
        "model_file": str(model_path),
        "variant": filprefix,
        "random_state": random_state,
        "selection_metric": metric,
        "metrics": best_row.to_dict(),
        "hparam_sok": bast_params,
    }

    if threshold is not None:
        meta["threshold"] = float(threshold)
        meta["threshold_rule"] = threshold_rule

    if utvarderings_grans is not None:
        meta["utvarderings_grans"] = float(utvarderings_grans)

    if vikter:
        meta["composite_vikter"] = {
            k: v for k, v in vikter.items() if k in
            ("w_mae", "w_high_mae", "w_high_bias", "min_antal_high")
        }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"Sparade metadata: {meta_path}")
    return model_path


def las_champion_metadata(models_dir, modell_filnamn):
    """
    Läser championens JSON-metadata — filen som delar namnstam med
    modellfilen (.joblib -> .json). Används av live-routern och den
    offline-utvärderingen för att läsa nisch-championens gräns för
    höga PM2.5 (plan 01), så att metadata är enda källan till gränsen.

    Parametrar
    ----------
    models_dir : Path
        Modellmappen.
    modell_filnamn : str
        Filnamnet ur champion-loggens 'sokvag'-kolumn (med eller utan
        .joblib-ändelse).

    Returnerar
    -------
    dict
        Metadatainnehållet, eller {} fail-säkert om filen saknas eller
        inte går att läsa.
    """
    namn = Path(modell_filnamn).name
    if namn.endswith(".joblib"):
        namn = namn[:-len(".joblib")]
    meta_path = Path(models_dir) / f"{namn}.json"

    if not meta_path.is_file():
        log.warning("Metadatafilen %s hittades inte.", meta_path)
        return {}

    try:
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        log.warning("Kunde inte läsa %s (%s).", meta_path, e)
        return {}
    


def append_champion_log(log_path, model_name, metric, score, sokvag=None, extra=None):
    # sokvag är en frivillig parameter. Om ingen sökväg skickas med, 
    # till exempel när baslinjen vinner, blir den tom.
    """
    Appendar en rad med vinnande modell, timestamp, metric och score
    till champion-loggen (CSV). Skapar filen med rubrikrad om den saknas.

    extra (plan 02): valfri dict med ytterligare kolumner per rad,
    t.ex. kompositens delvärden score_mae / score_high_mae /
    score_high_bias. Nya kolumner migreras in i den befintliga
    rubrikraden via _append_med_schema_migration, så att gamla rader
    (före plan 02) förblir läsbara med tomma fält.

    Parametrar
    ----------
    log_path : Path
        Sökväg till loggfilen, t.ex. "logs/champion_vanlig.csv".
    model_name : str
        Namn på vinnaren, t.ex. "Linear Regression" eller
        "Persistence Baseline" (om baslinjen vann räknas det som en giltig
        körning och loggas därmed också).
    metric : str
        Metrik som avgjorde valet, t.ex. "Composite" (plan 02) eller
        "Validation High PM2.5 MAE" (nisch-linjen, plan 01).
    score : float
        Vinnarens värde på metricken.
    sokvag : Path eller None
        Sökväg till championens .joblib-fil. Tom om baslinjen vann.
    extra : dict eller None
        Ytterligare kolumner för raden (plan 02).
    """
    timestamp = datetime.now().isoformat()
    rad = pd.DataFrame(
        [
            {
                "modell": model_name,
                "timestamp": timestamp,
                "metric": metric,
                "score": score,
                # Bara filnamnet sparas, så att loggen fungerar på alla datorer.
                # Modellen läses in med cfg.models_dir / filnamnet.
                "sokvag": Path(sokvag).name if sokvag is not None else "",
                **(extra or {}),
            }
        ]
    )

    # Rubrikraden migreras vid nya kolumner (plan 02), annars gäller
    # filens ordning och raden appenderas som vanligt.
    _append_med_schema_migration(Path(log_path), rad)
    print(f"Loggade champion i: {log_path} ({model_name}, {timestamp})")


def champion_is_fresh(log_path, antal_dagar, namn_suffix="") -> bool:
    """
    Avgör om senaste champion-loggningen för DENNA horisonten är gjord
    inom de senaste 'antal_dagar' dagarna.

    Med namn_suffix (t.ex. " (T+6)") räknas bara rader där modellnamnet
    slutar på suffixet — både riktiga modeller och baseline-vinster
    loggas med suffix. Rader från en annan horisont kan då aldrig hålla
    träningen tillbaka vid ett horisontbyte.

    Returnerar True om champion är färsk (orkestratorns steg 5 kan skippas),
    False annars. Fail-safe: om filen saknas, saknar matchande rader eller
    innehåller en ogiltig timestamp returneras False, så att en full
    träning tvingas fram.

    Parametrar
    ----------
    log_path : Path
        Sökväg till loggfilen, t.ex. "logs/champion_vanlig.csv".
    antal_dagar : int
        Maximal ålder i dagar för att champion ska räknas som färsk.
    namn_suffix : str
        Horisont-suffixet i modellnamnet, t.ex. " (T+6)". Tom sträng =
        filtrera inte (bakåtkompatibelt beteende).
    """
    log_path = Path(log_path)
    if not log_path.is_file():
        return False

    try:
        log_df = pd.read_csv(log_path)

        # Filtrera först på linjens horisont, kolla sedan åldern
        if namn_suffix:
            if "modell" not in log_df.columns:
                return False
            log_df = log_df[log_df["modell"].fillna("").str.endswith(namn_suffix)]

        if log_df.empty or "timestamp" not in log_df.columns:
            return False

        senaste_timestamp = pd.to_datetime(log_df["timestamp"].iloc[-1])
        if pd.isna(senaste_timestamp):
            return False

        alder = datetime.now() - senaste_timestamp.to_pydatetime()
        return alder <= timedelta(days=antal_dagar)

    except Exception as e:
        print(f"Varning: kunde inte läsa {log_path} ({e}) - kör full träning.")
        return False

    

def las_champion_via_sokvag(champion_log, models_dir):
    """
    Läser in den senaste sparade champion-modellen via kolumnen 'sokvag'
    i champion-loggen. Loggen är enda källan till vilken fil som är
    aktuell — aldrig filsystemets tidsstämplar eller filnamnsglobber,
    vilket blir osäkert när antalet sparade modellfiler växer.

    Rader där 'sokvag' är tom hoppas över. Det är rader där baslinjen
    vann, och då sparades ingen modell.

    Returnerar
    -------
    tuple
        (modell, namn, filnamn), eller (None, None, None) om loggen
        saknas, saknar kolumnen 'sokvag' eller om modellfilen inte finns.
    """
    log_path = Path(champion_log)
    if not log_path.is_file():
        log.info("Ingen champion-logg hittades: %s", log_path)
        return None, None, None

    try:
        log_df = pd.read_csv(log_path)
    except Exception as e:
        log.warning("Kunde inte läsa %s (%s).", log_path, e)
        return None, None, None

    if "sokvag" not in log_df.columns:
        log.warning("Loggen %s saknar kolumnen 'sokvag'.", log_path)
        return None, None, None

    # Behåll bara rader där en modell faktiskt sparades.
    # En tom sökväg blir NaN när CSV-filen läses in.
    modellrader = log_df[log_df["sokvag"].notna() & (log_df["sokvag"] != "")]
    if modellrader.empty:
        log.info("Ingen sparad champion-modell i %s.", log_path)
        return None, None, None

    # Den senaste raden är den senaste championen
    senaste = modellrader.iloc[-1]
    filnamn = senaste["sokvag"]
    modellfil = Path(models_dir) / filnamn

    if not modellfil.is_file():
        log.warning("Modellfilen finns inte: %s", modellfil)
        return None, None, None

    log.info("Läser in champion %s från %s", senaste["modell"], modellfil)
    return joblib.load(modellfil), senaste["modell"], filnamn


def hamta_champion_modell(models_dir, champion_log):
    """
    Laddar den senaste sparade champion-modellen (exklusive förskjuna
    varianter) och returnerar (modell, namn).

    Delegerar till las_champion_via_sokvag: kolumnen 'sokvag' i
    champion-loggen är den enda källan till vilken fil som är aktuell.
    Tidigare valdes filen med namnglob + filsystemets modifieringstid,
    vilket blev osäkert när antalet sparade modellfiler växer (städnings-
    logiken i stada_models_dir förutsätter också loggstyrd inläsning).

    Returnerar
    -------
    tuple
        (modell, namn) om en modell hittas, annars (None, None).
    """
    modell, namn, _ = las_champion_via_sokvag(champion_log, models_dir)
    return modell, namn


def _append_med_schema_migration(log_path, df):
    """
    Appendar rader till en CSV-logg, med hantering av nya kolumner.

    Skiljer sig den befintliga rubrikraden från df:s kolumner skrivs
    filen om EN gång med unionen av kolumnerna — gamla rader får tomma
    värden för de nya kolumnerna, och kolumnordningen blir alltid
    "gamla kolumner först, nya sist". Utan migreringen skulle rader
    med fler fält än rubriken hamna ur fas när pandas läser filen
    igen (t.ex. nisch_threshold i prediktioner_loggen, plan 01).

    Skapar filen med rubrikrad om den saknas.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if not log_path.is_file():
        df.to_csv(log_path, index=False)
        return

    befintlig = pd.read_csv(log_path)
    nya_kolumner = [k for k in df.columns if k not in befintlig.columns]

    if nya_kolumner:
        befintlig = befintlig.reindex(columns=[*befintlig.columns, *nya_kolumner])
        pd.concat([befintlig, df], ignore_index=True).to_csv(log_path, index=False)
    else:
        # Kolumnordningen i filen gäller — appenderade raders fält hamnar
        # alltid rätt även om df:n byggts upp i en annan ordning.
        df.reindex(columns=befintlig.columns).to_csv(
            log_path, mode="a", header=False, index=False,
        )


def append_jamfordelse_log(log_path, df):
    """
    Appendar en rad (eller flera) med head-to-head-jämförelse till en CSV.
    Skapar filen med rubrikrad om den saknas. Nya kolumner (t.ex.
    High-PM-mått från plan 01:s routade system) migreras in i den
    befintliga rubrikraden, se _append_med_schema_migration.
    """
    _append_med_schema_migration(log_path, df)
    print(f"Loggade jämförelse i: {log_path}")


def append_datakontroll_log(
    log_path, dataset, antal_merged, antal_validerad, antal_kvar, saknade,
    kolumnstatistik=None,
):
    """
    Skriver ut hur många rader som togs bort av dropna, och appendar
    samma information som en rad i datakontroll-loggen (CSV).
    Skapar filen med rubrikrad om den saknas.

    Plan 04: med kolumnstatistik (sammanstall_bortfall) loggas även det
    saknade datans MÖNSTER — längsta sammanhängande gap och antal gap
    >= horisont per kolumn — inte bara omfånget. dropna-policyn
    påverkas inte.

    Parametrar
    ----------
    log_path : Path
        Sökväg till loggfilen, t.ex. "logs/datakontroll_<par>.csv".
    dataset : str
        Namn på datasetet, t.ex. filnamnet för train eller val.
    antal_merged : int
        Antal rader i den mergade filen.
    antal_validerad : int
        Antal rader efter validate_data. Kan vara fler än antal_merged,
        eftersom saknade timmar läggs till som tomma rader.
    antal_kvar : int
        Antal rader efter dropna.
    saknade : pd.Series
        Antal saknade värden per feature/target-kolumn, före dropna.
        En rad kan sakna flera värden, så summan kan vara större än
        antalet borttagna rader.
    kolumnstatistik : pd.DataFrame, valfri
        Mönsterstatistik per kolumn ur sammanstall_bortfall (kolumn,
        langsta_gap_timmar, antal_gap_over_horisont). Kolumner med 0
        gap skrivs också, så loggen får samma kolumner vid varje körning.
    """
    log_path = Path(log_path)
    borttagna = antal_validerad - antal_kvar
    andel = borttagna / antal_validerad

    # Utskrift
    log.info("--- Datakontroll: %s ---", dataset)
    log.info("Rader i mergad fil: %s", antal_merged)
    log.info("Rader efter validate_data: %s", antal_validerad)
    log.info("Rader efter dropna: %s", antal_kvar)
    log.info("Borttagna rader: %s (%.1f%%)", borttagna, andel * 100)
    
    log.info("Saknade värden per kolumn (en rad kan sakna flera):\n%s",
         saknade[saknade > 0].to_string())

    # Plan 04: kompakt mönstersammanfattning i körloggen — topp 3
    # kolumner med längst sammanhängande gap.
    if kolumnstatistik is not None and not kolumnstatistik.empty:
        topp = kolumnstatistik.sort_values(
            "langsta_gap_timmar", ascending=False
        ).head(3)
        log.info(
            "Längsta sammanhängande data-gap per kolumn (topp 3):\n%s",
            topp[["kolumn", "langsta_gap_timmar", "antal_gap_over_horisont"]]
            .to_string(index=False),
        )

    # En rad i loggen. Alla kolumner tas med, även de med 0 saknade,
    # så att loggen har samma kolumner vid varje körning.
    rad = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset": dataset,
        "rader_merged": antal_merged,
        "rader_validerad": antal_validerad,
        "rader_kvar": antal_kvar,
        "borttagna": borttagna,
        "andel_borttagna": round(andel, 4),
    }
    for kolumn, antal in saknade.items():
        rad[f"saknade_{kolumn}"] = int(antal)

    # Plan 04: mönsterkolumner per kolumn (0 för rena kolumner — samma
    # princip som saknade_<kolumn> ovan).
    if kolumnstatistik is not None:
        for _, stat in kolumnstatistik.iterrows():
            rad[f"langsta_gap_{stat['kolumn']}_h"] = int(
                stat["langsta_gap_timmar"]
            )
            rad[f"antal_gap_over_horisont_{stat['kolumn']}"] = int(
                stat["antal_gap_over_horisont"]
            )

    # Schema-migrering (plan 04): de nya mönsterkolumnerna får inte
    # hamna ur fas med äldre rubrikrader i befintliga per-parsloggar —
    # gamla rader får tomma fält för de nya kolumnerna.
    _append_med_schema_migration(log_path, pd.DataFrame([rad]))
    log.info("Loggade datakontroll i: %s", log_path)


def append_bortfallsanalys_log(log_path, dataset, gaprader):
    """
    Plan 04: skriver en rad per sammanhängande data-gap >= 6 h till
    logs/bortfallsanalys_<par>.csv — med start, slut, längd och månad
    blir dagsblocken (t.ex. SMHI-refuserad nederbörd) och långa
    pm25-gap synliga och spårbara över tid, utan att dropna-policyn
    ändras. Gap-tabellen lever utanför datakontroll-loggen eftersom
    antalet rader varierar per körning (0 till flera).

    Appendar INGENTING när gaprader är tom — ett rent dataset ska inte
    generera brus; datakontroll-raden bevisar att kontrollen kördes
    (där är langsta_gap-kolumnerna 0).

    Appendar via schema-migrering, så nya kolumner aldrig hamnar ur fas
    med en befintlig rubrikrad.

    Parametrar
    ----------
    log_path : Path
        Sökväg till loggfilen, t.ex. "logs/bortfallsanalys_<par>.csv".
    dataset : str
        Namn på datasetet som gapen hör till (fylls i som en kolumn).
    gaprader : pd.DataFrame
        En rad per gap (kolumn, start, slut, langd_timmar, manad) ur
        sammanstall_bortfall. Tom/None => inget skrivs.
    """
    if gaprader is None or gaprader.empty:
        return

    rad_df = gaprader.copy()
    rad_df.insert(0, "dataset", dataset)
    rad_df.insert(0, "timestamp", datetime.now().isoformat(timespec="seconds"))

    _append_med_schema_migration(log_path, rad_df)
    log.info("Loggade %d data-gap i %s.", len(rad_df), log_path)


# Används av predict pipeline och live.py
def append_skippad_log(
    log_path, orsak, t_obs=None, linje=None,
    pm25_alder_h=None, alinje="hela körningen",
):
    """
    Plan 04: en maskinläsbar rad per hoppad live-körning (eller linje),
    så att "varför är det så få prediktioner?" går att besvara på en
    blick och uppföljningens stickprov blir ärligt istället för
    tystande krympande.

    En NORMAL körning (inga överhopp) skriver ingenting — loggen ska
    innehålla exakt en rad per verkligt hoppat försök.

    Parametrar
    ----------
    log_path : Path
        Sökväg till loggfilen, t.ex. "logs/live_skippad_<par>.csv".
    orsak : str
        Orsak enligt plan 04:s enum: ingen_ankarpunkt,
        for_gammal_ankarpunkt, t_obs_redan_predikterad,
        prognos_saknas_maltid, champion_saknas:<linje>,
        vader_slot_ogiltig:<linje> eller ovantat_fel:<Typ> (live.py:s
        fångst-som-sista-nät).
    t_obs : pd.Timestamp, valfri
        Ankarpunkten om den hann bestämmas (annars tom).
    linje : str, valfri
        Berörd linje när en ENKEL linje hoppades över.
    pm25_alder_h : float, valfri
        T_obs ålder i timmar om den är känd.
    alinje : str
        "hela körningen" (alla linjer) eller "linje" (enstaka linje).

    Kolumner: timestamp, t_obs, orsak, linje, alinje, pm25_alder_h.
    Skapar filen med rubrikrad om den saknas. Appendar via
    schema-migrering, så nya kolumner aldrig hamnar ur fas med en
    befintlig rubrikrad.
    """
    rad = pd.DataFrame([{
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "t_obs": t_obs.isoformat() if t_obs is not None else "",
        "orsak": orsak,
        "linje": linje if linje is not None else "",
        "alinje": alinje,
        "pm25_alder_h": (
            "" if pm25_alder_h is None else round(float(pm25_alder_h), 2)
        ),
    }])
    _append_med_schema_migration(log_path, rad)


# Används av predict pipeline
def append_prediktioner_log(log_path, poster_df):
    """Appendar live-prediktioner (en rad per linje) till loggen.

    Skapar filen med rubrikrad om den saknas — samma mönster som
    append_champion_log. Nya kolumner migreras in i den befintliga
    rubrikraden (plan 01: nisch_threshold), se
    _append_med_schema_migration.
    """
    _append_med_schema_migration(log_path, poster_df)


def stada_models_dir(models_dir, linjer, keep=2):
    """
    Tar bort champion-modellfiler i models_dir som inte längre behövs,
    så att mappen inte växer i all evighet.

    För varje modellinje behålls de `keep` senaste loggade raderna med
    modellfil (aktuell champion + fallback-generat(n)) — allt annat som
    matchar projektets filsuffix (.joblib/.json) tas bort. Champion-
    loggen (kolumnen 'sokvag') är enda källan till vilka filer som är
    aktuella, vilket las_champion_via_sokvag garanterar.

    Parametrar
    ----------
    models_dir : Path
        Modellmappen som ska städas. Finns den inte görs inget.
    linjer : list[tuple[Path, str]]
        En (champion_logg, filprefix)-tupel per modellinje, t.ex.
        [(cfg.champion_log, ""), (cfg.forskjuten_log, "forskjuten_")].
    keep : int
        Antal modellfiler per linje som behålls.

    Fail-safe: saknas en linjes logg, eller går den inte att läsa,
    raderas inga filer för den linjen — vi kan då inte veta vilka filer
    som är aktuella. Prefixet "" (vanliga linjen) matchar alla filnamn,
    så saknas den vanliga champion-loggen städas ingenting alls.

    Returnerar
    -------
    list[str]
        Namnen på de borttagna filerna.
    """
    models_dir = Path(models_dir)
    if not models_dir.is_dir():
        return []

    behalla = set()          # filnamn som ska sparas
    okanda_prefix = []        # linjer vars logg inte kunde läsas

    for log_path, filprefix in linjer:
        log_path = Path(log_path)
        if not log_path.is_file():
            okanda_prefix.append(filprefix)
            continue
        try:
            log_df = pd.read_csv(log_path)
        except Exception as e:
            log.warning(
                "Kunde inte läsa %s (%s) — städar inte den linjens filer.",
                log_path, e,
            )
            okanda_prefix.append(filprefix)
            continue
        if "sokvag" not in log_df.columns:
            okanda_prefix.append(filprefix)
            continue

        modellfiler = log_df[log_df["sokvag"].notna() & (log_df["sokvag"] != "")]
        behalla.update(modellfiler["sokvag"].tail(keep))

    # .json-metadatafilen delar namnstam med .joblib-filen: behåll båda.
    for namn in list(behalla):
        if namn.endswith(".joblib"):
            behalla.add(namn[: -len(".joblib")] + ".json")

    borttagna = []
    for fil in models_dir.iterdir():
        if fil.suffix not in (".joblib", ".json"):
            continue
        if fil.name in behalla:
            continue
        # Linje vars logg inte kunde läsas: rör inte dess filer.
        if any(fil.name.startswith(prefix) for prefix in okanda_prefix):
            continue

        try:
            fil.unlink()
            borttagna.append(fil.name)
        except OSError as e:
            log.warning("Kunde inte ta bort %s (%s).", fil, e)

    if borttagna:
        log.info(
            "Städade %s: tog bort %d modellfil(er): %s",
            models_dir, len(borttagna), ", ".join(sorted(borttagna)),
        )
    return borttagna


def trim_prediktioner_log(prediktioner_log, linje, behall_modell):
    """
    Kastar live-prediktioner för `linje` som INTE gjordes med någon av
    modellerna i `behall_modell`. Körs när en ny champion sparats:
    prediktionernas långsiktiga roll är uppföljning per champion-era, och
    när championen byts behövs den gamla erans rader inte längre.

    Med nisch-routingen (plan 01) delar linjen vanlig på eror från
    BÅDE den ordinarie championen och nisch-championen — därför tar
    behall_modell emot en lista (eller ett enskilt namn, bakåtkompatibelt)
    och championbyten på endera sidan behåller den andras rader.

    Rader för de andra linjerna lämnas orörda, liksom rader som gjordes
    med någon av modellerna som ska behållas. Rubrikraden bevaras och
    omskrivningen är atomär (temporär fil som ersätter den gamla).

    NOTERING: orealiserade prediktioner från den gamla championen för-
    svinner vid ett championbyte — en uppföljning direkt efter ett byte
    täcker bara den nya eran. Med dagens låga championbyte-takt är
    det ett medvetet val.

    Parametrar
    ----------
    prediktioner_log : Path
        Sökväg till prediktionsloggen.
    linje : str
        Vilken linje som trimmas, t.ex. "vanlig".
    behall_modell : str eller lista[str]
        Modellnamn(al) vars rader för linjen behålls.

    Returnerar
    -------
    int
        Antal borttagna rader (0 om loggen saknas eller inget behövde
        trimmas).
    """
    behall_modeller = [behall_modell] if isinstance(behall_modell, str) else list(behall_modell)

    log_path = Path(prediktioner_log)
    if not log_path.is_file():
        return 0

    logg = pd.read_csv(log_path)
    if "linje" not in logg.columns or "modell" not in logg.columns:
        log.warning(
            "Prediktionsloggen %s saknar kolumner — ingen trim utförs.",
            log_path,
        )
        return 0
    if logg.empty:
        return 0

    behalla = (logg["linje"] != linje) | logg["modell"].isin(behall_modeller)
    antal_borttagna = int((~behalla).sum())
    if antal_borttagna == 0:
        return 0

    temporar = log_path.with_name(log_path.name + ".tmp")
    logg[behalla].to_csv(temporar, index=False)
    temporar.replace(log_path)

    log.info(
        "Trimmade %s: tog bort %d rad(er) för linjen %s — behållna modeller: %s.",
        log_path, antal_borttagna, linje, ", ".join(behall_modeller),
    )
    return antal_borttagna