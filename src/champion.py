from pathlib import Path
from src.metrics import evaluate_model, display_df, composite_score

import joblib
import pandas as pd

from src.io import las_champion_via_sokvag
from src.logger import setup_logger

log = setup_logger(__name__)


def load_latest_champion(log_path, models_dir):
    """
    Läser in den senast sparade championen via kolumnen 'sokvag' i loggen.

    Delegerar till las_champion_via_sokvag (src/io.py), så att champion-
    inläsningen har EN enda implementering för båda modellinjerna:
    loggen — inte filsystemet — avgör vilken fil som är aktuell. Båda
    städningen av models_dir och inläsningen måste använda samma regel.

    Rader där 'sokvag' är tom hoppas över. Det är rader där baslinjen
    vann, och då sparades ingen modell.

    Returnerar (modell, namn, filnamn), eller (None, None, None) om
    loggen saknas, saknar kolumnen 'sokvag' eller om modellfilen inte finns.
    """
    return las_champion_via_sokvag(log_path, models_dir)


def compete_with_champion(
    new_model,
    new_name,
    old_model,
    old_name,
    data_path,
    features,
    target,
    threshold=None,
    vikter=None,
):
    """
    Låter dagens bästa kandidat och den tidigare championen göra
    prognoser på samma VALIDERINGSdata (plan 02), utan att träna om
    någon av dem.

    Avgörs av den viktade kompositscoren (se composite_score i
    src/metrics.py): MAE + High-PM MAE + |High-PM Bias|, med samma
    gräns och vikter som kandidatvalet. Testdata används ALDRIG i valet
    — det låsta test-snapshottet i data/test/last är enbart en
    oberoende slutrapport.

    Dagens modell vinner bara om den är STRIKT bättre (lägre komposit).
    Vid lika resultat behålls den tidigare championen. Om det inte
    finns någon tidigare champion vinner dagens modell automatiskt.

    Parametrar
    ----------
    new_model, new_name: kandidaten ur turneringen och dess namn.
    old_model, old_name : befintlig champion (None => kandidaten vinner).
    data_path : Path
        Sökväg till processad VALIDERINGSdata (cfg.val_processed
        respektive den förskjutna varianten).
    features : list[str]
        Feature-kolumner (cfg.features).
    target : str
        Målkolumn (cfg.target_column).
    threshold : float, valfri
        Gräns för High-PM-måtten — p90 av träningsmålet, beräknad en
        gång i run_model_line och återanvänd här och i turneringen.
    vikter : dict, valfri
        Vikter till kompositscoren ([model.composite] i config.toml).

    Returnerar
    -------
    tuple
        (winner, results_df), där winner är "new" eller "old".
    """
    # Läs in valideringsdata
    data_df = pd.read_csv(data_path)
    X = data_df[features]
    y = data_df[target]

    # Utvärdera den nya kandidaten (med gräns, så kompositen blir
    # jämförbar med turneringens siffror)
    new_results = evaluate_model(y, new_model.predict(X), threshold=threshold)
    new_score = composite_score(new_results, vikter)

    # Om ingen tidigare champion finns
    if old_model is None:
        log.info(
            "Ingen tidigare champion finns. %s vinner automatiskt.",
            new_name
        )

        results_df = pd.DataFrame([
            {"Modell": f"{new_name} (dagens bästa)", **new_results,
             "Composite": new_score}
        ])

        display_df(results_df)
        return "new", results_df

    # Utvärdera tidigare champion på samma valideringsdata
    old_results = evaluate_model(y, old_model.predict(X), threshold=threshold)
    old_score = composite_score(old_results, vikter)

    # Strikt lägre komposit vinner; vid lika behålls den tidigare
    # championen (beslut 2026-10-03:s tie-regel).
    new_wins = new_score < old_score

    # Sammanställ duelresultaten
    results_df = pd.DataFrame([
        {"Modell": f"{new_name} (dagens bästa)", **new_results,
         "Composite": new_score},
        {"Modell": f"{old_name} (tidigare champion)", **old_results,
         "Composite": old_score},
    ])

    display_df(results_df)

    winner = "new" if new_wins else "old"

    log.info(
        "Champion-duell på valideringsdata (%s rader), kompositscore: "
        "dagens %.4f, tidigare %.4f. Vinnare: %s.",
        len(data_df), new_score, old_score, winner,
    )

    return winner, results_df