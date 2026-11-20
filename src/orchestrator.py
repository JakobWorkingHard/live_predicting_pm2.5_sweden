import pandas as pd

from src.config import ProjectConfig
from src.metrics import jamfor_championer, jamfor_routat_system, display_df
from src.logger import setup_logger
from src.champion import load_latest_champion
from src.data_pipeline import fetch_and_merge, fetch_and_merge_test, build_processed_datasets, las_last_test_vagar
from src.model_pipeline import run_model_line, run_nisch_linje
from src.io import champion_is_fresh, stada_models_dir, las_champion_metadata

log = setup_logger(__name__)


def orchestration(cfg: ProjectConfig):
    log.info("Startar projektets orchestration.")

    # Suffix i champion-namnen. Horisonten (" (T+6)") finns på BÅDA linjerna
    # — det är målets förskjutning och champion_is_fresh:s skydd mot tysta
    # horisontbyten (filtret matchar på att namnet slutar med suffixet).
    # Linjerna markeras symmetriskt: "SVR vanlig (T+6)" resp.
    # "SVR forskjuten (T+6)". Stavningen "forskjuten" följer projektets
    # konvention (forskjuten_log, filprefix) och håller filnamnen ASCII-rena.
    vanlig_suffix = f" vanlig (T+{cfg.horisont})"
    forskjuten_suffix = f" forskjuten (T+{cfg.horisont})"
    # Nisch-linjen (plan 01): samma horisontsiffra, egen etikett —
    # champion_is_fresh matchar endast nisch-rader.
    nisch_suffix = f" nisch (T+{cfg.horisont})"

    # Endast den VANLIGA championen styr: är den färsk (tränad inom
    # antal_dagar_sedan) körs inte pipelinen alls — då visas bara det
    # senaste jämförelseresultatet ur jamfordelse-loggen (se else-grenen).
    # Fail-safe i champion_is_fresh: saknas logg, matchande rad eller
    # giltig timestamp returneras False, så att en full träning tvingas fram.
    if not champion_is_fresh(
        cfg.champion_log,
        cfg.antal_dagar_sedan,
        namn_suffix=vanlig_suffix,
    ):
        # 1. TESTDATA: Hämta och sammanfoga (rullande latest-months).
        # Sedan plan 02 används testdata aldrig i något modellval —
        # hämtningen driver datakontrollen/provensiens och det bearbetade
        # test-datasetet; den oberoende rapporten kör mot det LÅSTA
        # test-snapshottet i data/test/last (block 6).
        testvagar = fetch_and_merge_test(cfg)

        # 2. Hämta och sammanfoga data
        fetch_and_merge(
            cfg,
            cfg.train_start,
            cfg.train_slut,
            cfg.train_pm25,
            cfg.train_vader,
            cfg.train_merged,
            "Träningsdata",
        )

        fetch_and_merge(
            cfg,
            cfg.validering_start,
            cfg.validering_slut,
            cfg.val_pm25,
            cfg.val_vader,
            cfg.val_merged,
            "Valideringsdata",
        )

        # 3. VALIDERING, FEATURE ENGINEERING OCH SPARANDE
        build_processed_datasets(cfg, testvagar=testvagar)

        # 4. VANLIG MODELLINJE
        # Nisch-championens live-prediktioner tillhör samma linje
        # ("vanlig") i prediktionsloggen — vid ett byte av den ordinarie
        # championen ska nisch-erans rader inte försvinna (plan 01).
        _, nisch_namn_fore, _ = load_latest_champion(cfg.nisch_log, cfg.models_dir)

        run_model_line(
            cfg,
            line_label="vanlig",
            train_path=cfg.train_processed,
            val_path=cfg.val_processed,
            log_path=cfg.champion_log,
            antal_dagar=cfg.antal_dagar_sedan,
            name_suffix=vanlig_suffix,
            extra_behall_modeller=[nisch_namn_fore] if nisch_namn_fore else None,
        )

        # 4b. NISCH-LINJEN (höga PM2.5, plan 01): turnering enbart på
        # rader med pm25 >= p90(tränings-pm25); vinnaren möter den
        # ordinarie championen head-to-head på de filtrerade
        # valideringsraderna och antas bara vid strikt bättre
        # High PM2.5 MAE. Vid adoption servar nisch-championen den
        # vanliga linjens live-prediktioner vid höga halter
        # (se predict_pipeline); annars servas allt av championen.
        # Körs på sin egen färskhetsklocka, precis som de andra linjerna.
        run_nisch_linje(
            cfg,
            train_path=cfg.train_processed,
            val_path=cfg.val_processed,
            antal_dagar=cfg.antal_dagar_sedan,
            name_suffix=nisch_suffix,
        )

        # 5. FÖRSKJUTEN MODELLINJE (måltidsväder, T + horisont)
        # Vädret på varje rad ersätts med observationen vid T + horisont.
        # OBS för framtiden: vid live-prediktion ska väder-slotten fyllas med
        # en väderPROGNOS för måltiden (T + horisont), inte en observation
        # (se docs/evaluation.md).

        forskjuten = run_model_line(
            cfg,
            line_label="förskjuten",
            train_path=cfg.train_processed_forskjuten,
            val_path=cfg.val_processed_forskjuten,
            log_path=cfg.forskjuten_log,
            antal_dagar=cfg.forskjuten_antal_dagar_sedan,
            name_suffix=forskjuten_suffix,
            filprefix="forskjuten_",
        )

        # 6. Jämför de två championerna på det LÅSTA test-snapshottet
        # (plan 02, informationellt): vanlig vs förskjuten champion plus
        # persistence-baseline, på de testrader som är kompletta i BÅDA
        # varianterna. Snapshottet i data/test/last skapas med
        # "python main.py --ladda-test" och används ALDRIG i något val —
        # championduellen avgörs på valideringsdata (se run_model_line).
        # Saknas snapshottet hoppas jämförelsen över (fail-safe).
        if forskjuten is not None:
            last_vagar = las_last_test_vagar(cfg)
            if last_vagar is None:
                log.warning(
                    "Inget låst test-snapshot i %s — head-to-head-jämförelsen "
                    "hoppas över. Skapa det med: python main.py --ladda-test",
                    cfg.test_last_dir,
                )
            else:
                forskjuten_modell, forskjuten_namn = forskjuten
                champion_modell, champion_namn, _ = load_latest_champion(
                    cfg.champion_log,
                    cfg.models_dir,
                )
                jamfor_championer(
                    cfg,
                    champion_modell=champion_modell,
                    champion_namn=champion_namn,
                    forskjuten_modell=forskjuten_modell,
                    forskjuten_namn=forskjuten_namn,
                    test_path_vanlig=last_vagar["test_processed"],
                    test_path_forskjuten=last_vagar["test_processed_forskjuten"],
                )

        # 6b. ROUTAT SYSTEM vs ORDINARIE CHAMPION (plan 01,
        # informationellt): syns endast när en nisch-champion finns —
        # annars pågår ingen rutning. Gränsen läses ur nisch-championens
        # metadata (enda källan), och jämförelsen körs på samma
        # valideringsdata som adoptionsduellen. Körs oavsett om
        # nisch-linjen tränade om den här körningen: routing från en
        # tidigare era lever kvar tills en duell avsätter den.
        nisch_modell, nisch_namn, nisch_fil = load_latest_champion(
            cfg.nisch_log, cfg.models_dir,
        )
        if nisch_modell is not None:
            nisch_grans = las_champion_metadata(cfg.models_dir, nisch_fil).get("threshold")
            champion_modell, champion_namn, _ = load_latest_champion(
                cfg.champion_log, cfg.models_dir,
            )
            if nisch_grans is not None and champion_modell is not None:
                jamfor_routat_system(
                    cfg,
                    champion_modell=champion_modell,
                    champion_namn=champion_namn,
                    niche_modell=nisch_modell,
                    niche_namn=nisch_namn,
                    niche_threshold=nisch_grans,
                    val_path=cfg.val_processed,
                )
            elif nisch_grans is None:
                log.warning(
                    "Nisch-championen %s saknar gräns i metadata — "
                    "rutad system-jämförelse hoppas över.", nisch_namn,
                )
    else:
        # Vanlig champion är färsk: hoppa över HELA pipelinen (ingen
        # datahämtning, ingen omträning) och visa det senaste
        # jämförelseresultatet ur logs/jamforelse_champion.csv.
        # En head-to-head-körning loggas med tre rader (champion,
        # förskjuten modell, baseline) som delar samma timestamp.
        log.info(
            "Den vanliga championen är fortfarande färsk (gräns: %s dagar). "
            "Pipelinen hoppas över — visar senaste jämförelseresultat.",
            cfg.antal_dagar_sedan,
        )

        if cfg.jamfordelse_log.is_file():
            jamdf = pd.read_csv(cfg.jamfordelse_log)
            senaste_tid = jamdf["timestamp"].iloc[-1]
            display_df(jamdf[jamdf["timestamp"] == senaste_tid])
        else:
            log.info(
                "Ingen tidigare jämförelse finns i %s — inget att visa.",
                cfg.jamfordelse_log,
            )

    # 7. STÄDNING AV MODELLMAPPEN (båda grenarna: efter träning såväl som
    # vid överhoppad träning). Behåll aktuell champion + en fallback per
    # modellinje — nisch-linjen är den tredje (plan 01) — och ta bort
    # övriga sparade modellfiler, annars växer models_dir i all evighet.
    # Fail-säkert: saknas en linjes logg raderas inga filer för den
    # linjen (se stada_models_dir).
    stada_models_dir(
        cfg.models_dir,
        [
            (cfg.champion_log, ""),
            (cfg.forskjuten_log, "forskjuten_"),
            (cfg.nisch_log, "nisch_"),
        ],
        keep=2,
    )
