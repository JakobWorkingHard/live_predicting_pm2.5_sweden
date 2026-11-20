import argparse
from pathlib import Path

import pandas as pd

from src.config import ladda_baskonfig, bygg_station_config
from src.stations import hamta_stationpar
from src.orchestrator import orchestration
from src.data_pipeline import skapa_last_test_snapshot
from src.metrics import display_df
from src.logger import setup_logger

log = setup_logger(__name__)

# Räkna ut projektets rotkatalog dynamiskt
PROJEKT_ROT = Path(__file__).resolve().parent


def valj_stationpar(stationpar, filter_str):
    """
    Filtrerar parslistan på namn (skiftlägesokänslig delsträng i PM2.5-
    eller väderstationsnamnet) när --station används. Ex:
        python main.py --station "Malmö"
    """
    if not filter_str:
        return stationpar

    trafna = [
        par for par in stationpar
        if filter_str.lower() in par.pm25_namn.lower()
        or filter_str.lower() in par.vaderstation_namn.lower()
    ]
    if not trafna:
        raise SystemExit(f"Ingen station matchade '{filter_str}'.")
    log.info("Filtrerade fram %d par på '%s'.", len(trafna), filter_str)
    return trafna


def main():
    parser = argparse.ArgumentParser(
        description=(
            "PM2.5-projektets pipeline: tränar och utvärderar modeller "
            "för ALLA upptäckta stationpar (PM2.5-station + närmaste "
            "väderstation)."
        )
    )
    parser.add_argument(
        "--ladda-test",
        action="store_true",
        help=(
            "Plan 02: hämtar den aktuella testperioden från SMHI:s "
            "latest-months och frysar den som ett LÅST test-snapshot per "
            "stationpar i data/test/last_<tidsserie_id> (manifest + "
            "processade varianter). Används ENBART för oberoende "
            "rapportering — aldrig modellval. Oberoende av championernas "
            "färskhetsklockor."
        ),
    )
    parser.add_argument(
        "--ersatt",
        action="store_true",
        help=(
            "Ersätt ett befintligt låst test-snapshot (engångsjobb, "
            "dokumentera om-snapshottet i docs/decisions.md först)."
        ),
    )
    parser.add_argument(
        "--station",
        type=str,
        default=None,
        help=(
            "Kör endast par vars PM2.5- eller väderstationsnamn innehåller "
            "denna sträng (t.ex. --station \"Malmö\"). Utan flaggan körs "
            "alla par."
        ),
    )
    args = parser.parse_args()

    log.info("Startar PM2.5-projektets pipeline")

    # 1. Bas-konfiguration (ren TOML — inga API-anrop) ...
    bas = ladda_baskonfig(PROJEKT_ROT / "config.toml", PROJEKT_ROT)
    log.info("Baskonfigurationen har laddats")

    # 2. ... sedan stationparen: färsk cache läses, annars körs en full
    # upptäckt mot SMHIs API:er (ENDAST här — inget sker vid import).
    stationpar = hamta_stationpar(bas)
    stationpar = valj_stationpar(stationpar, args.station)

    log.info(
        "Kör %d stationpar: %s",
        len(stationpar),
        ", ".join(p.pm25_namn for p in stationpar),
    )

    # 3. Engångsjobbet --ladda-test fryser test-snapshot för varje par.
    # Eget fel per par loggas men stoppar inte resten.
    if args.ladda_test:
        resultat = []
        for par in stationpar:
            try:
                cfg = bygg_station_config(bas, par)
                skapades = skapa_last_test_snapshot(cfg, ersatt=args.ersatt)
                resultat.append(
                    (par.pm25_namn, "skapade" if skapades else "fanns/redan")
                )
            except Exception as e:
                log.error("Test-snapshot för %s misslyckades: %s", par.pm25_namn, e)
                resultat.append((par.pm25_namn, f"FEL: {e}"))
        _visa_sammanfattning(resultat, "Test-snapshot")
        return

    # 4. Huvudflödet: full pipeline per par. Ett par som failar (t.ex.
    # SMHI-datagap eller API-fel) loggas och resten kör vidare.
    resultat = []
    for par in stationpar:
        try:
            cfg = bygg_station_config(bas, par)
            orchestration(cfg)
            resultat.append((par.pm25_namn, par.vaderstation_namn, "OK"))
        except Exception as e:
            log.error(
                "Stationparet %s + %s misslyckades: %s",
                par.pm25_namn, par.vaderstation_namn, e,
            )
            resultat.append((par.pm25_namn, par.vaderstation_namn, f"FEL: {e}"))

    _visa_sammanfattning(resultat, "Pipeline")
    log.info("PM2.5-projektets pipeline har slutförts")


def _visa_sammanfattning(resultat, rubrik):
    """Skriver en slutlig status-tabell: ett par per rad."""
    if not resultat:
        return
    kolumner = (
        ["pm25_station", "status"] if len(resultat[0]) == 2
        else ["pm25_station", "vaderstation", "status"]
    )
    df = pd.DataFrame(resultat, columns=kolumner)
    ok = int((~df["status"].str.startswith("FEL")).sum())

    print(f"\n=== Sammanfattning: {rubrik} ===")
    display_df(df)
    log.info("%d av %d par lyckades.", ok, len(df))


if __name__ == "__main__":
    main()
