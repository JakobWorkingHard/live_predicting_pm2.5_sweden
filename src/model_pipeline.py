import pandas as pd 

from src.io import ( 
    champion_is_fresh, 
    save_champion_model, 
    append_champion_log, 
    trim_prediktioner_log,
    las_champion_metadata, ) 

from src.train_models import ( 
    load_train_val_data, 
    build_model_registry, 
    sok_parametrar_for_registry,
    train_and_evaluate_models, 
    persistence_baseline,
    berakna_high_pm_threshold,
    filtrera_high_pm,
    PersistenceModel, )

from src.metrics import ( 
    select_best_model, 
    build_comparison_table, 
    display_df, evaluate_model,
    composite_score, ) 
from src.champion import load_latest_champion, compete_with_champion 

from src.logger import setup_logger 

log = setup_logger(__name__)


def _validation_row(model, name, X_val, y_val, threshold=None, vikter=None): 
    """Beräknar tidigare champions resultat på valideringsdata
    för att kunna visa det tillsammans med övriga valideringsresultat.
    
    Plan 02: raden innehåller samma mått som turneringens rader —
    inklusive High-PM-måtten och den viktade kompositscoren — så att
    champion-loggen alltid kan redovisa delvärden, vilken champion
    som än behålls."""
    
    resultat = evaluate_model(y_val, model.predict(X_val), threshold=threshold)
    
    rad = {"Modell": name}
    for nyckel, varde in resultat.items():
        rad[f"Validation {nyckel.replace('R2', 'R²')}"] = varde
    rad["Validation Composite"] = composite_score(resultat, vikter)
    return pd.Series(rad)


def run_model_line( cfg, line_label, train_path, val_path, log_path, antal_dagar, name_suffix="", filprefix="", extra_behall_modeller=None):

    """ Kör hela modellflödet för en modellinje (vanlig eller förskjuten):
    1. Hoppar över allt om championen (med linjens horisont-suffix i
    namnet) är yngre än antal_dagar. Den färska championen laddas ändå
    in och returneras, så att orkestratorns jämförelse kan köras även
    utan omträning.
    2. Tränar alla modeller och jämför dem med baslinjen på valideringsdatan.
    3. Låter dagens bästa modell tävla mot den tidigare championen på
    VALIDERINGSDATAN (plan 02), avgjort av den viktade kompositscoren.
    Den tidigare championen behålls om den är minst lika bra.
    4. Loggar vinnaren i log_path, med sökvägen till modellfilen.

    Plan 02: testdatan används INTE i något val — funktionen tar ingen
    test_path-parameter alls längre; det låsta test-snapshottet
    (data/test/last, "python main.py --ladda-test") är enbart en
    oberoende rapport.

    extra_behall_modeller: ytterligare modellnamn vars rader för linjen
    behålls vid championbyte (trim_prediktioner_log). Plan 01: den
    vanliga linjens prediktioner delas mellan den ordinarie championen
    och nisch-championen — orkestratorn skickar nisch-championens namn
    hit, så att ett ordinarie championbyte inte raderar nisch-eran.

    Returnerar (champion_modell, champion_namn) om linjen tränades och en
    modell blev champion, samt när linjen hoppades över därför att en
    befintlig champion med sparad modellfil var färsk.
    Returnerar None om baslinjen vann den här körningen, eller om den
    färska championen saknar sparad modellfil (t.ex. baseline-vinst). """
 
    # 1. Är championen fortfarande färsk? 
    # Suffixet kopplar färskheten till rätt horisont — en " (T+1)"-rad
    # kvalar aldrig in som färsk " (T+6)"-champion.
    if champion_is_fresh(log_path, antal_dagar, namn_suffix=name_suffix): 
        champion_log = pd.read_csv(log_path) 
        log.info( "En befintlig %s champion är fortfarande aktuell. " 
                 "Tidpunkt: %s | Modell: %s | Mått: %s | Score: %s", 
                 line_label,
                 champion_log["timestamp"].iloc[-1], 
                 champion_log["modell"].iloc[-1], 
                 champion_log["metric"].iloc[-1],
                 champion_log["score"].iloc[-1], ) 
        
        # Ladda den färska championen även om inget tränades om, så att
        # orkestratorns head-to-head-jämförelse kan visas vid varje körning.
        champion_modell, champion_namn, _ = load_latest_champion( 
            log_path, cfg.models_dir, ) 
        
        if champion_modell is None: 
            log.info( 
                "Den färska %s-championen har ingen sparad modellfil "
                "(t.ex. baseline-vinst). Inget att jämföra.", 
                line_label, 
            ) 
            return None 
        
        log.info( 
            "Läser in den färska %s-championen %s för jämförelsen.", 
            line_label, champion_namn, 
        ) 
        return champion_modell, champion_namn 
 
    log.info( "Ingen tillräckligt aktuell %s champion finns. " 
             "Startar modellträning. Gräns: %s dagar.", 
             line_label, 
             antal_dagar, 
             )

    # 2. Träna och utvärdera alla modeller på valideringsdatan 
    X_train, y_train, X_val, y_val = load_train_val_data(train_path, 
                                                         val_path, 
                                                         cfg.features, 
                                                         cfg.target_column, 
                                                         ) 
    
    # 2a. Utvärderingsgräns för High-PM-måtten (plan 02): p90 av
    # TRÄNINGSMÅLET (y_train — värdet vid T+horisont). Beräknas EN gång
    # här och delas av turnering, baslinje och championduell, så att
    # kandidatval och duell alltid poängsätter samma siffror.
    # OBS skilj den från nisch-linjens ROUTNINGS-gräns (p90 av pm25
    # vid T, se berakna_high_pm_threshold i src/train_models.py) —
    # två gränser med två olika roller (se docs/evaluation.md).
    utvarderings_grans = float(y_train.quantile(0.90))
    log.info(
        "Utvärderingsgräns för High-PM-mått (p90 av träningsmålet): "
        "%.2f µg/m³", utvarderings_grans,
    )

    # Plan 02: kompositvikterna ([model.composite]); getattr-mönstret
    # gör linjen körbar även för minimala cfg (t.ex. tester).
    composite_vikter = getattr(cfg, "composite_vikter", None)

    # 2b. Hyperparametersökning (plan 05, beslut 2026-11-15): körs
    # ENDAST här, vid omträning (championens 7-dagarsklocka) – aldrig
    # i live-prediktioner. Sökningen sker på träningsperioden med
    # TimeSeriesSplit(gap=horisont); validerings- och testdata används
    # inte av sökningen. Resultatet delas mellan modellinjerna inom
    # samma körning (se sok_parametrar_for_registry) och misslyckade
    # familjer behåller sina standardparametrar (fail-safe).
    bast_params = None
    if getattr(cfg, "hparam_aktiverad", False) and getattr(cfg, "hparam_grids", None):
        bast_params = sok_parametrar_for_registry(
            X_train,
            y_train,
            grids=cfg.hparam_grids,
            horisont=cfg.horisont,
            n_splits=cfg.hparam_n_splits,
            lambda_high_pm=cfg.hparam_lambda_high_pm,
            sok_rader_per_modell={"SVR": cfg.hparam_svr_max_rader},
            models_dir=cfg.models_dir,
        )
    
    all_models = build_model_registry(cfg.random_state, bast_params=bast_params) 
    results_df,trained_models = train_and_evaluate_models(
        all_models,
        X_train,
        y_train,
        X_val,
        y_val,
        high_pm25_threshold=utvarderings_grans,
        vikter=composite_vikter,
        )
    
    log.info("Träningen av den %s modellinjen är klar.", line_label) 
    
    # Jämför modellerna med baslinjen. build_comparison_table sorterar 
    # redan tabellen så att den bästa modellen hamnar överst. 
    
    baseline_train, baseline_val = persistence_baseline( X_train, y_train, X_val,y_val,
                                                        high_pm25_threshold=utvarderings_grans, )
    comparison_df = build_comparison_table( results_df, baseline_val, cfg.candidate_selection_metric,
                                            vikter=composite_vikter, )
    
    display_df(comparison_df) 
    
    best_name, best_row = select_best_model( comparison_df, metric=cfg.candidate_selection_metric, )
    
    log.info( "Bästa alternativ (%s): %s. Urvalsmått: %s. Score: %.4f.", 
             line_label, 
             best_name, 
             cfg.candidate_selection_metric,
             best_row[cfg.candidate_selection_metric],
             )
    
    champion_namn = f"{best_name}{name_suffix}"
    champion_modell = None

    # 3. Vem blir champion?
    if best_name == "Persistence Baseline":
        # Ingen modell slog baslinjen (plan 02: baslinjens komposit
        # ingår ärligt i jämförelsen). Ingen modell sparas, så det
        # finns ingen sökväg att logga.
        log.info("Ingen modell slog baseline (%s). Sparar ingen champion.", line_label)

        champion_row = best_row
        modell_sokvag = None

    else:
        # Hämta modellens resultat för metadata
        full_row = results_df[results_df["Modell"] == best_name].iloc[0]
        # Låt dagens bästa modell tävla mot den tidigare championen på
        # VALIDERINGSDATAN (plan 02), avgjort av kompositscoren med
        # samma gräns och vikter som turneringen. Den tidigare
        # championen tränas inte om.

        old_model, old_name, old_file = load_latest_champion(log_path, cfg.models_dir, )

        winner, duel_results_df = compete_with_champion(
            new_model=trained_models[best_name],
            new_name=champion_namn,
            old_model=old_model,
            old_name=old_name,
            data_path=val_path,
            features=cfg.features,
            target=cfg.target_column,
            threshold=utvarderings_grans,
            vikter=composite_vikter, )

        if winner == "new":
             # Dagens modell vann och sparas som ny champion.
             # # save_champion_model returnerar sökvägen till .joblib-filen.

             champion_row = full_row
             champion_modell = trained_models[best_name]

             modell_sokvag = save_champion_model( model=champion_modell,
                                                 model_name=champion_namn,
                                                 best_row=full_row,
                                                 models_dir=cfg.models_dir,
                                                 random_state=cfg.random_state,
                                                 metric=cfg.champion_selection_metric,
                                                 filprefix=filprefix,
                                                 bast_params=(bast_params or {}).get(best_name),
                                                 # Plan 02: score-regeln ska
                                                 # gå att utläsa ur metadatan.
                                                 vikter=composite_vikter,
                                                 utvarderings_grans=utvarderings_grans, ) 
             
             log.info("Champion %s har sparats.", champion_namn)

             # Championbyte: den gamla championens live-prediktioner
             # behövs inte längre (uppföljning sker per champion-era) —
             # trimma prediktionsloggen till den nya eran. filprefix
             # mappar mot linjevärdena i prediktionsloggen:
             # "forskjuten_" -> "forskjuten", annars "vanlig".
             # extra_behall_modeller skyddar nisch-championens rader
             # för den vanliga linjen (plan 01).
             linje = "forskjuten" if filprefix else "vanlig"
             trim_prediktioner_log(
                 cfg.prediktioner_log,
                 linje=linje,
                 behall_modell=[champion_namn, *(extra_behall_modeller or [])],
             )

        else:
            # Den tidigare championen vann. Ingen ny modell sparas, men 
            # championen loggas igen med dagens datum, så att
            # champion_is_fresh inte tränar om vid varje körning. 
            # Hit kommer vi bara om det fanns en tidigare champion, 
            # eftersom compete_with_champion annars returnerar "new". 
            
            assert old_model is not None
            champion_namn = old_name
            champion_modell = old_model
            modell_sokvag = old_file
            champion_row = _validation_row(
                old_model, old_name, X_val, y_val,
                threshold=utvarderings_grans,
                vikter=composite_vikter,
            )
            
            log.info("Den tidigare championen %s behålls.", old_name) 
            
    log.info("Slutligt valt alternativ (%s): %s",line_label, champion_namn) 
    
    display_df(champion_row.to_frame().T)


    # 4. Logga vinnaren (plan 02): kompositscoren + delvärdena, så att
    # varje loggrad kan tolkas i efterhand ("vilken score-regel
    # producerade raden?"). champion_row har alltid Validation-kolumnerna
    # — oavsett om vinnaren är kandidaten (full_row/turneringen),
    # baslinjen (comparison-raden) eller den gamla championen
    # (_validation_row).
    logged_metric = "Composite"
    logged_score = champion_row["Validation Composite"]

    # Delvärden till kompositscoren (nan om High-PM saknas i valideringen).
    extra_falt = {
        "score_mae": champion_row["Validation MAE"],
        "score_high_mae": champion_row["Validation High PM2.5 MAE"],
        "score_high_bias": champion_row["Validation High PM2.5 Bias"],
    }

    append_champion_log(
        log_path=log_path,
        model_name=champion_namn,
        metric=logged_metric,
        score=logged_score,
        sokvag=modell_sokvag,
        extra=extra_falt,
    )

    log.info(
        "Champion-loggen (%s) har uppdaterats. Modell: %s, mått: %s, score: %.4f.",
        line_label,
        champion_namn,
        logged_metric,
        logged_score,
    )

    
    if champion_modell is None: 
        return None 
    
    return champion_modell, champion_namn


 
 
 


#----------------------------------------------------------
# Nisch-linjen för höga PM2.5-värden (plan 01, beslut 2026-11-15)
#----------------------------------------------------------

# Minsta antal träningsrader för att nisch-turneringen ska vara
# meningsfull (p90 av ~21 000 rader ≈ 2 100 rader; under ~100 rader
# blir modellerna och gränsen alltför osäkra).
_NISCH_MIN_TRADER = 100


def run_nisch_linje(cfg, train_path, val_path, antal_dagar, name_suffix, extra_behall_modeller=None):
    """
    Kör nisch-modellinjen för höga PM2.5-halter (plan 01):

    1. Hoppar över träningen om den befintliga nisch-championen (med
       linjens suffix) är färsk — modellen laddas istället och
       returneras tillsammans med gränsen ur sin metadata.
    2. Beräknar gränsen (berakna_high_pm_threshold: p90 av tränings-
       pm25, dvs. värdet vid T — samma kolumn routern nycklar på) och
       tränar en turnering ENBART på rader med pm25 >= gränsen.
       Persistence-baseline ingår (plan 01: vinner baslinjen är det
       ett giltigt utfall — den sparas då som PersistenceModel).
    3. Adoption sker bara om kandidaten, i tur och ordning:
       a) har lägst "Validation High PM2.5 MAE" i nisch-turneringen,
       b) slår den ORDINARIE championen STRIKT på samma mått på samma
          filtrerade valideringsrader (lika eller sämre -> ingen
          routing; NaN -> ingen routing),
       c) slår en eventuell tidigare nisch-champion strikt (samma
          tie-regel som de ordinarie linjerna: den gamla behålls).
    4. Avvisade dueller loggas INTE i champion_nisch.csv (beslut
       2026-11-15): duellen körs om vid nästa pipeline-körning, och en
       avvisad tidsstämpel kan aldrig blockera omvärdering efter att
       den ordinarie championen bytts.

    ALLA jämförelser (filter, turnering, båda duellerna) använder
    SAMMA kolumn (pm25 vid T) och SAMMA gräns — ingen mismatch.
    Medvetet ingen hyperparametersökning här (plan 01:s risknotering:
    underlaget är för knappt; plan 05 söker bara för de ordinarie
    linjerna).

    Returnerar
    -------
    tuple
        (modell, namn, threshold) när en nisch-champion finns (ny
        sparad eller färsk inläst), annars (None, None, None).
    """
    nisch_metric = "Validation High PM2.5 MAE"

    # Plan 02: vikterna används bara för komposit-KOLUMNEN (redovisning)
    # — aldrig för antagningsregeln. getattr möjliggör minimala cfg.
    composite_vikter = getattr(cfg, "composite_vikter", None)

    # 1. Är den befintliga nisch-championen färsk? (Egen 7-dagars-
    # klocka, samma mönster som de ordinarie linjerna.)
    if champion_is_fresh(cfg.nisch_log, antal_dagar, namn_suffix=name_suffix):
        modell, namn, filnamn = load_latest_champion(cfg.nisch_log, cfg.models_dir)
        threshold = las_champion_metadata(cfg.models_dir, filnamn).get("threshold")

        if modell is not None and threshold is not None:
            log.info(
                "En befintlig nisch champion är fortfarande aktuell: %s "
                "(gräns %.2f µg/m³). Ingen nisch-omträning.",
                namn, threshold,
            )
            return modell, namn, threshold

        log.info(
            "Den färska nisch-championen saknar modellfil eller gräns i "
            "metadata — en full nisch-turnering tvingas fram."
        )

    log.info(
        "Ingen tillräckligt aktuell nisch champion finns. Startar "
        "nisch-turnering för höga PM2.5-värden. Färskhetsgräns: %s dagar.",
        antal_dagar,
    )

    # 2. Läs in data; gränsen beräknas på FULLA träningsdata innan
    # filtreringen, så att filter, turnering och router aldrig skiljer sig.
    X_train, y_train, X_val, y_val = load_train_val_data(
        train_path, val_path, cfg.features, cfg.target_column,
    )

    threshold = berakna_high_pm_threshold(X_train)
    log.info("Gräns för höga PM2.5 (p90 av tränings-pm25): %.2f µg/m³", threshold)

    X_tr, y_tr, X_v, y_v = filtrera_high_pm(
        X_train, y_train, X_val, y_val, threshold,
    )

    log.info(
        "Nisch-turnering: %d träningsrader, %d valideringsrader med pm25 över gränsen.",
        len(X_tr), len(X_v),
    )

    # High-måttet (mask på målet y >= gräns) måste vara definierat på de
    # filtrerade valideringsraderna, annars kan duellen inte avgöras.
    if len(X_tr) < _NISCH_MIN_TRADER or len(X_v) == 0 or not bool((y_v >= threshold).any()):
        log.warning(
            "För få nisch-rader (train %d, val %d) eller inga valrader med "
            "målet över gränsen — nisch-linjen hoppas över den här körningen.",
            len(X_tr), len(X_v),
        )
        return None, None, None

    # 3. Turneringen, enbart på de filtrerade raderna. Kompositscoren
    # (plan 02) beräknas också här, men avgör ALDRIG nisch-linjen — den
    # redovisas bara vid duellen, så att era-jämförelser förblir
    # tolkbara. Nisch-linjens regel förblir High PM2.5 MAE.
    all_models = build_model_registry(cfg.random_state)
    results_df, trained_models = train_and_evaluate_models(
        all_models, X_tr, y_tr, X_v, y_v, high_pm25_threshold=threshold,
        vikter=composite_vikter,
    )

    baseline_train, baseline_val = persistence_baseline(
        X_tr, y_tr, X_v, y_v, high_pm25_threshold=threshold,
    )

    comparison_df = build_comparison_table(results_df, baseline_val, nisch_metric)
    display_df(comparison_df)

    best_name, best_row = select_best_model(comparison_df, metric=nisch_metric)
    kandidat_score = best_row[nisch_metric]

    log.info(
        "Bästa nisch-kandidat (%s): %s. Score: %.4f.",
        nisch_metric, best_name, kandidat_score,
    )

    champion_namn = f"{best_name}{name_suffix}"

    if best_name == "Persistence Baseline":
        # Vinner baslinjen nisch-turneringen är det ett giltigt utfall
        # (plan 01): vid höga halter är "nuvärdet består" idag det
        # bästa. Wrappas som PersistenceModel så att den kan sparas och
        # routas precis som vilken champion som helst.
        niche_modell = PersistenceModel()
    else:
        niche_modell = trained_models[best_name]

    # 4a. Duel mot den ordinarie championen på SAMMA filtrerade
    # valideringsrader — precis dit routern skulle skicka prediktioner.
    ovr_modell, ovr_namn, _ = load_latest_champion(cfg.champion_log, cfg.models_dir)
    if ovr_modell is None:
        log.warning(
            "Ingen ordinarie champion-modell att möta — nisch-championen "
            "kan inte antas den här körningen."
        )
        return None, None, None

    ovr_metrics = evaluate_model(y_v, ovr_modell.predict(X_v), threshold=threshold)
    ovr_score = ovr_metrics["High PM2.5 MAE"]

    duell_df = pd.DataFrame([
        {
            "Modell": champion_namn,
            "High PM2.5 MAE": kandidat_score,
            "High PM2.5 Bias": best_row.get("Validation High PM2.5 Bias"),
            "High PM2.5 Count": best_row.get("Validation High PM2.5 Count"),
            # Plan 02: kompositen redovisas vid sidan av — aldrig som —
            # nisch-linjens antagningsregel.
            "Composite": best_row.get("Validation Composite"),
        },
        {
            "Modell": f"{ovr_namn} (ordinarie champion)",
            "High PM2.5 MAE": ovr_metrics["High PM2.5 MAE"],
            "High PM2.5 Bias": ovr_metrics["High PM2.5 Bias"],
            "High PM2.5 Count": ovr_metrics["High PM2.5 Count"],
            "Composite": composite_score(ovr_metrics, composite_vikter),
        },
    ])
    display_df(duell_df)

    if pd.isna(kandidat_score) or pd.isna(ovr_score) or not (kandidat_score < ovr_score):
        log.info(
            "Nisch-kandidaten slog inte den ordinarie championen på %s "
            "(%.4f vs %.4f) — ingen routing: den ordinarie championen servar allt.",
            nisch_metric, kandidat_score, ovr_score,
        )
        return None, None, None

    # 4b. Duel mot en eventuell tidigare nisch-champion. Alla tre
    # jämförs på DAGENS filtrerade rader med DAGENS gräns — samma
    # underlag för alla är det rättvist sättet att jämföra eror.
    gammal_modell, gammal_namn, _ = load_latest_champion(cfg.nisch_log, cfg.models_dir)
    if gammal_modell is not None:
        gammal_score = evaluate_model(
            y_v, gammal_modell.predict(X_v), threshold=threshold,
        )["High PM2.5 MAE"]
        if pd.isna(gammal_score) or not (kandidat_score < gammal_score):
            log.info(
                "Tidigare nisch-champion %s behålls (%s: %.4f vs %.4f).",
                gammal_namn, nisch_metric, kandidat_score, gammal_score,
            )
            return None, None, None

    # 5. Adoption: spara med gränsen i metadata, trimma gamla nisch-eror
    # ur prediktionsloggen och logga championen.
    modell_sokvag = save_champion_model(
        model=niche_modell,
        model_name=champion_namn,
        best_row=best_row,
        models_dir=cfg.models_dir,
        random_state=cfg.random_state,
        metric=nisch_metric,
        filprefix="nisch_",
        bast_params=None,
        threshold=threshold,
        threshold_rule=cfg.high_pm_threshold_rule,
    )
    log.info("Nisch-champion %s har sparats och antagits.", champion_namn)

    # Era-trim: den vanliga linjens prediktioner hålls per era för BÅDE
    # den ordinarie championen och nisch-championen (routing delar linjen).
    trim_prediktioner_log(
        cfg.prediktioner_log,
        linje="vanlig",
        behall_modell=[champion_namn, ovr_namn, *(extra_behall_modeller or [])],
    )

    append_champion_log(
        log_path=cfg.nisch_log,
        model_name=champion_namn,
        metric=nisch_metric,
        score=kandidat_score,
        sokvag=modell_sokvag,
    )

    log.info(
        "Champion-loggen (%s) har uppdaterats. Modell: %s, mått: %s, score: %.4f.",
        cfg.nisch_log, champion_namn, nisch_metric, kandidat_score,
    )

    return niche_modell, champion_namn, threshold
