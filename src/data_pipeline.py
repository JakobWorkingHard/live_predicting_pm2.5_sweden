# Hämtar PM2.5 samt väder och slår ihop dem. 
# Merged används på både train och val 

import json
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.extracting_data import hamta_pm25, vader_data, hamta_testperiod, hamta_test_vader
from src.config import skapa_testvagar
from src.io import (
    generate_csv,
    merge_df_and_generate_csv,
    append_datakontroll_log,
    append_bortfallsanalys_log,
)
from src.logger import setup_logger 
from src.validate_process import validate_data, sammanstall_bortfall
from src.features import ( 
    add_wind_features, 
    add_time_features,
    ersatt_vader_med_maltidsvader, 
    )
 

log = setup_logger(__name__)

def fetch_and_merge(cfg, start, end, pm25_path, vader_path, merged_path, label): 
    """ Hämtar PM2.5 och väderdata för en period och slår ihop dem. Används för både tränings- 
    och valideringsdata. Om filerna redan finns läses de istället för att hämtas på nytt. label 
    används bara i loggmeddelandena, t.ex. "Träningsdata". """

    #Hämta PM2.5 
    pm_data = hamta_pm25(start, 
                         end, 
                         cfg.pm25_tidsserie_id, 
                         cfg.luft_api, pm25_path, 
                         )
    
    generate_csv(pm_data, pm25_path) 
    log.info("%s för PM2.5 har hämtats och sparats.", label)
 
    # Hämta väder 
    vader = vader_data( 
                cfg.vader_parametrar, 
                start, 
                end, 
                cfg.vaderstation_id, 
                cfg.vader_api, 
                vader_path, 
                )
    
    generate_csv(vader, vader_path) 
    
    log.info("%s för väder har hämtats och sparats.", label)

    # Slå ihop PM2.5 och väder 
    merge_df_and_generate_csv( 
        pm25_path, 
        vader_path, 
        "tid", 
        merged_path, 
        False, 
        )
    
    log.info("%s för PM2.5 och väder har sammanfogats.", label)





def fetch_and_merge_test(cfg): 
    """ Hämtar PM2.5 och väderdata för den dynamiska testperioden och slår ihop dem. 
    Testperioden går från första tidpunkten i SMHI:s latest-months fram till nu minus 
    fyra timmar. Returnerar testvagar, en dict med sökvägarna till testfilerna —
    och sedan plan 02 även perioden själv ("test_start"/"test_slut"), som bl.a.
    det låsta test-snapshottets manifest använder. Testdatan används sedan
    plan 02 ALDRIG i något modellval; den rullande hämtningen driver bara
    datakontrollen och processed-bygget (proveniens, plan 04/06). """
    
    # Beräkna testperioden och sökvägarna till testfilerna 
    test_start, test_slut = hamta_testperiod(
    station_id=cfg.vaderstation_id,
    VADER_API=cfg.vader_api,
    validering_slut=cfg.validering_slut,
    
    ) 
    
    log.info("Dynamisk testperiod: %s till %s", test_start, test_slut)
    
    testvagar = skapa_testvagar(cfg, test_start, test_slut)

    # Plan 02: perioden följer med i dicten (främst till det låsta
    # test-snapshottets manifest, se skapa_last_test_snapshot).
    testvagar["test_start"] = pd.Timestamp(test_start).strftime("%Y-%m-%d %H:%M")
    testvagar["test_slut"] = pd.Timestamp(test_slut).strftime("%Y-%m-%d %H:%M")

    # Hämta PM2.5 
    test_pm_data = hamta_pm25( 
        test_start, 
        test_slut, 
        cfg.pm25_tidsserie_id, 
        cfg.luft_api,testvagar["test_pm25"], 
        ) 
    
    generate_csv(test_pm_data, testvagar["test_pm25"]) 
    
    log.info("PM2.5-testdata har hämtats och sparats.")
 
    # Hämta väder (latest-months) 
    test_vader_data = hamta_test_vader( 
        parametrar=cfg.vader_parametrar,
        test_start=test_start, 
        test_slut=test_slut, 
        station_id=cfg.vaderstation_id, 
        VADER_API=cfg.vader_api, 
        )
    
    generate_csv(test_vader_data, testvagar["test_vader"]) 
    
    log.info("Vädertestdata har hämtats och sparats.")

    # Slå ihop PM2.5 och väder 
    merge_df_and_generate_csv( 
        testvagar["test_pm25"], 
        testvagar["test_vader"], 
        "tid",
        testvagar["test_merged"], 
        False, 
        ) 
    
    log.info("Testdata för PM2.5 och väder har sammanfogats.") 
    return testvagar


def build_processed_dataset(cfg, merged_df, output_path, mal_vader):
    """
    Bearbetar ett sammanslaget dataset: validering, features, target och
    dropna. Loggar en datakontroll och sparar resultatet i output_path.
    mal_vader=True innebär att väderkolumnerna ersätts med måltidsväder:
    vädrets värde på rad T kommer från T + horisont, dvs. när PM2.5-målet
    realiseras. Används för den förskjutna modellinjen.
    """

    log.info("Bearbetar dataset: %s", output_path.name)
 
    antal_merged = len(merged_df)
 
    # Validera data
    df = validate_data(merged_df)
    antal_validerad = len(df)
 
    if mal_vader:
        # Vindriktning skiftas FÖRE add_wind_features, så att vind_sin/
        # vind_cos räknas ur måltids-vindriktningen. Skiftet använder
        # alltid horisonten: måltiden är T + horisont, samma som mål-
        # skiftet, så vädret och målet kan aldrig hamna ur fas.

        df = ersatt_vader_med_maltidsvader(
            df,
            cfg.forskjuten_vaderkolumner,
            cfg.horisont,
        )
 
    # Skapa features
    df = add_wind_features(df)
    df = add_time_features(df, cfg.horisont)
 
    # Skapa målvariabel: PM2.5 ett antal timmar framåt
    df[cfg.target_column] = df["pm25"].shift(-cfg.horisont)
 
    # Behåll endast kompletta modellrader
    modellkolumner = list(cfg.features) + [cfg.target_column]
    saknade = df[modellkolumner].isna().sum()

    # Plan 04: mönsterstatistiken MÅSTE samlas in FÖRE dropna — efteråt
    # är raderna borta och gaps/block kan inte längre rekonstrueras.
    # validate_data har garanterat en komplett timaxel, så ett NaN är
    # ett verkligt gap. dropna-policyn påverkas inte (beslut 2026-11-15).
    kolumnstatistik, gaprader = sammanstall_bortfall(
        df, modellkolumner, cfg.horisont,
    )

    df = df.dropna(subset=modellkolumner)

    # Logga datakontroll — plan 04: med mönsterstatistiken (längsta gap,
    # antal gap >= horisont) per kolumn.
    append_datakontroll_log(
        cfg.datakontroll_log,
        output_path.name,
        antal_merged,
        antal_validerad,
        len(df),
        saknade,
        kolumnstatistik=kolumnstatistik,
    )

    # Plan 04: en rad per gap >= 6 h i den separata bortfallsanalysen —
    # skriver ingenting när datasetet är rent (0-gap bevisas av
    # datakontrollens kolumner istället).
    append_bortfallsanalys_log(cfg.bortfallsanalys_log, output_path.name, gaprader)
 
    # Spara bearbetat dataset
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)

    generate_csv(df, output_path, index=False)

    log.info("Bearbetat dataset har sparats: %s", output_path.name)
 
 
def build_processed_datasets(cfg, testvagar):
    """
    Bygger de sex bearbetade dataseten: train, val och test, både för den
    vanliga linjen och för den förskjutna linjen (måltidsväder).
    """

    # Läs in de sammanfogade dataseten
    train_merged_df = pd.read_csv(cfg.train_merged)
    val_merged_df = pd.read_csv(cfg.val_merged)
    test_merged_df = pd.read_csv(testvagar["test_merged"])
 
    # (sökväg att spara till, måltidsväder?) -> sammanslaget dataset

    merged_data = {
        (cfg.train_processed, False): train_merged_df,
        (cfg.val_processed, False): val_merged_df,
        (testvagar["test_processed"], False): test_merged_df,
        (cfg.train_processed_forskjuten, True): train_merged_df,
        (cfg.val_processed_forskjuten, True): val_merged_df,
        (testvagar["test_processed_forskjuten"], True): test_merged_df,
    }
 
    for (output_path, mal_vader), merged_df in merged_data.items():
        build_processed_dataset(cfg, merged_df, output_path, mal_vader)


#----------------------------------------------------------
# Låst test-snapshot (plan 02, beslut 2026-11-15)
#----------------------------------------------------------

# Filnamn i snapshot-katalogen (fasta namn — inga datum, filen ÄR
# daterad via manifestet i stället).
_LAST_TEST_FILNAMN = {
    "manifest": "manifest.json",
    "test_processed": "test_processed.csv",
    "test_processed_forskjuten": "test_processed_forskjuten.csv",
    "test_merged": "test_merged.csv",
}


def las_last_test_vagar(cfg):
    """
    Returnerar sökvägarna till det låsta test-snapshottet (plan 02) om
    det är komplett — manifest + båda processade varianterna — annars
    None. Snapshottet används ENBART av den oberoende rapporteringen
    (jamfor_championer, orchestrator block 6); inget automatiserat
    modellval läser det.

    Returnerar
    -------
    dict[str, Path] eller None
    """
    last_dir = getattr(cfg, "test_last_dir", None)
    if last_dir is None:
        return None

    last_dir = Path(last_dir)
    vagor = {
        "last_dir": last_dir,
        "manifest": last_dir / _LAST_TEST_FILNAMN["manifest"],
        "test_processed": last_dir / _LAST_TEST_FILNAMN["test_processed"],
        "test_processed_forskjuten": (
            last_dir / _LAST_TEST_FILNAMN["test_processed_forskjuten"]
        ),
    }

    if not all(
        vagor[namn].is_file()
        for namn in ("manifest", "test_processed", "test_processed_forskjuten")
    ):
        return None

    return vagor


def skapa_last_test_snapshot(cfg, ersatt=False):
    """
    Plan 02: hämtar den aktuella testperioden från latest-months och
    fryser den som ett LÅST test-snapshot under data/test/last — för
    oberoende utvärdering av championerna, ALDRIG för modellval.
    Körs som engångsjobb via "python main.py --ladda-test" (med
    " --ersatt" för ett dokumenterat om-snapshot).

    Policy (docs/decisions.md): om-snapshotta högst en gång per
    större version, och notera alltid beslutet i beslutsloggen —
    upprepade dueller mot samma testperiod är precis det adaptive-
    overfitting-risker som planen tar bort.

    Manifestet registrerar PM2.5-tidsserie, väderstation, period,
    hämtningstidpunkt och att källan är latest-months (preliminär,
    ej korrektionsgranskad data).

    Returnerar
    -------
    bool
        True om ett (nytt) snapshot skrevs, False om ett redan fanns
        och ersatt inte begärdes.
    """
    last_dir = Path(cfg.test_last_dir)
    last_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = last_dir / _LAST_TEST_FILNAMN["manifest"]

    if manifest_path.is_file() and not ersatt:
        log.error(
            "Ett låst test-snapshot finns redan (%s). Ett om-snapshot är "
            "ett versionsbeslut: dokumentera det i docs/decisions.md och "
            "kör igen med --ersatt.",
            manifest_path,
        )
        return False

    # Rensa ett eventuell tidigare snapshot (endast med --ersatt)
    for filnamn in _LAST_TEST_FILNAMN.values():
        fil = last_dir / filnamn
        if fil.is_file():
            fil.unlink()

    testvagar = fetch_and_merge_test(cfg)

    # Sammanfogningsfilen kopieras med — den gör snapshotet
    # reproducerbart utan nya API-anrop.
    shutil.copy2(
        testvagar["test_merged"],
        last_dir / _LAST_TEST_FILNAMN["test_merged"],
    )

    # De två processade varianterna byggs direkt i snapshot-katalogen:
    # vanligt väder (radens T) och måltidsväder (T + horisont).
    test_merged_df = pd.read_csv(
        last_dir / _LAST_TEST_FILNAMN["test_merged"]
    )
    ut_vanlig = last_dir / _LAST_TEST_FILNAMN["test_processed"]
    ut_forskjuten = last_dir / _LAST_TEST_FILNAMN["test_processed_forskjuten"]

    build_processed_dataset(cfg, test_merged_df, ut_vanlig, False)
    build_processed_dataset(cfg, test_merged_df, ut_forskjuten, True)

    manifest = {
        "hamtad_tidpunkt": datetime.now().isoformat(timespec="seconds"),
        "kalla": (
            "SMHI latest-months (preliminär data, ej korrektionsgranskad)"
        ),
        "test_period": {
            "start": testvagar.get("test_start"),
            "slut": testvagar.get("test_slut"),
        },
        "pm25_station": cfg.pm25_station,
        "pm25_tidsserie_id": cfg.pm25_tidsserie_id,
        "vaderstation": cfg.vaderstation,
        "vaderstation_id": cfg.vaderstation_id,
        "horisont": cfg.horisont,
        "anvandning": (
            "ENBART oberoende rapportering (jamfor_championer) — ALDRIG "
            "modellval (plan 02)"
        ),
        "filer": {
            "test_processed": ut_vanlig.name,
            "test_processed_forskjuten": ut_forskjuten.name,
            "test_merged": _LAST_TEST_FILNAMN["test_merged"],
        },
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    log.info(
        "Låst test-snapshot skapat i %s (period %s – %s).",
        last_dir,
        manifest["test_period"]["start"],
        manifest["test_period"]["slut"],
    )
    return True
 
