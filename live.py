import argparse
from pathlib import Path

from src.config import ladda_baskonfig, bygg_station_config
from src.io import append_skippad_log
from src.stations import hamta_stationpar
from src.logger import setup_logger
from src.predict_pipeline import kor_live_prediktion, folj_uppa

log = setup_logger(__name__)

# Räkna ut projektets rotkatalog dynamiskt
PROJEKT_ROT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="Live-prediktioner för PM2.5 (alla stationpar)")
    parser.add_argument(
        "--uppföljning",
        action="store_true",
        help="Rätta loggade prediktioner mot verkliga mätningar (steg 11)",
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

    log.info("Startar live-prediktion")

    # Bas-konfig + stationpar: cache läses om den är färsk, annars körs
    # upptäckten mot API:erna. Sker ENDAST här — inget sker vid import.
    bas = ladda_baskonfig(PROJEKT_ROT / "config.toml", PROJEKT_ROT)
    stationpar = hamta_stationpar(bas)

    if args.station:
        stationpar = [
            par for par in stationpar
            if args.station.lower() in par.pm25_namn.lower()
            or args.station.lower() in par.vaderstation_namn.lower()
        ]
        if not stationpar:
            raise SystemExit(f"Ingen station matchade '{args.station}'.")

    log.info("Kör live-flödet för %d stationpar.", len(stationpar))

    # Ett par som misslyckas (API-fel, ingen champion, datagap) loggas
    # och stoppar inte övriga par.
    lyckade = 0
    for par in stationpar:
        cfg = None
        try:
            cfg = bygg_station_config(bas, par)
            if args.uppföljning:
                folj_uppa(cfg)
            else:
                kor_live_prediktion(cfg)
            lyckade += 1
        except Exception as e:
            # Plan 04: även oväntade fel ska vara maskinläsbara — en rad
            # (orsak=ovantat_fel:<Typ>) i PARETS skipp-logg, så att felen
            # syns i uppföljningens skip-sammanfattning. Skapandet av
            # cfg (stationparskonfiguration) kraschade kan ingen rad
            # skrivas; misslyckas själva skip-radsskrivningen sväljs
            # det — den får aldrig döda loopen för övriga par.
            log.error("Live-körning för %s misslyckades: %s", par.pm25_namn, e)
            if cfg is not None:
                try:
                    append_skippad_log(
                        cfg.skippad_log,
                        orsak=f"ovantat_fel:{type(e).__name__}",
                    )
                except Exception:
                    log.warning(
                        "Kunde inte skriva skip-rad för %s.", par.pm25_namn,
                    )

    log.info("Live-körningen är klar: %d av %d par lyckades.", lyckade, len(stationpar))


if __name__ == "__main__":
    main()
