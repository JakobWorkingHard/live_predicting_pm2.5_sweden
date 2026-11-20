"""Tester för src/io.py: champion-färskhet per linje/horisont, champion-
inläsning via 'sokvag', städning av models_dir, era-trim av
prediktionsloggen, metadata för nisch-gränsen och schema-migrering av
append-loggar (plan 01)."""

from datetime import datetime, timedelta

import joblib
import pandas as pd

from src.io import (
    champion_is_fresh,
    hamta_champion_modell,
    las_champion_metadata,
    save_champion_model,
    stada_models_dir,
    trim_prediktioner_log,
    append_jamfordelse_log,
    append_prediktioner_log,
    append_champion_log,
    append_datakontroll_log,
    append_bortfallsanalys_log,
    append_skippad_log,
)


SUFFIX = " vanlig (T+6)"


def _skriv_championlogg(path, rader):
    pd.DataFrame(rader).to_csv(path, index=False)


def test_champion_is_fresh_saknad_fil(tmp_path):
    """Fail-safe: ingen logg -> False -> full träning tvingas fram."""
    assert champion_is_fresh(tmp_path / "finns_inte.csv", 7, SUFFIX) is False


def test_champion_is_fresh_med_matchande_suffix(tmp_path):
    logg = tmp_path / "champion.csv"
    _skriv_championlogg(logg, [
        {"modell": "SVR vanlig (T+6)",
         "timestamp": (datetime.now() - timedelta(days=1)).isoformat()},
    ])
    assert champion_is_fresh(logg, 7, SUFFIX) is True


def test_champion_is_fresh_ignorerar_annan_horisont(tmp_path):
    """Rader från en annan horisont håller inte träningen tillbaka."""
    logg = tmp_path / "champion.csv"
    _skriv_championlogg(logg, [
        {"modell": "SVR vanlig (T+1)", "timestamp": datetime.now().isoformat()},
    ])
    assert champion_is_fresh(logg, 7, SUFFIX) is False


def test_champion_is_fresh_ignorerar_forskjuten_linje(tmp_path):
    """En färsk forskjuten-champion håller inte vanlig-linjen tillbaka.

    Linjerna delar horisontsiffra men skiljer sig på etiketten i
    suffixet. Matchningen är endswith, så " vanlig (T+6)" får aldrig
    träffa "SVR forskjuten (T+6)" — inte ens om loggarna blandas.
    """
    logg = tmp_path / "champion.csv"
    _skriv_championlogg(logg, [
        {"modell": "SVR forskjuten (T+6)", "timestamp": datetime.now().isoformat()},
    ])
    assert champion_is_fresh(logg, 7, SUFFIX) is False


def test_champion_is_fresh_baseline_med_suffix_raknas(tmp_path):
    """Baseline loggas med suffix — en baseline-vinst räknas som färsk,
    så att linjen inte tränas om vid varje körning."""
    logg = tmp_path / "champion.csv"
    _skriv_championlogg(logg, [
        {"modell": f"Persistence Baseline{SUFFIX}",
         "timestamp": datetime.now().isoformat()},
    ])
    assert champion_is_fresh(logg, 7, SUFFIX) is True


def test_champion_is_fresh_gammal_champion(tmp_path):
    logg = tmp_path / "champion.csv"
    _skriv_championlogg(logg, [
        {"modell": "SVR vanlig (T+6)",
         "timestamp": (datetime.now() - timedelta(days=10)).isoformat()},
    ])
    assert champion_is_fresh(logg, 7, SUFFIX) is False


def test_hamta_champion_modell_hittar_suffixerad_champion(tmp_path):
    """Inläsningen styrs AV LOGGEN (kolumnen 'sokvag'), inte av filnamns-
    globbar: baseline-raden är senast loggad men har ingen modellfil och
    ska uteslutas, och filnamnet behöver inte härledas ur modellnamnet."""
    logg = tmp_path / "champion_vanlig.csv"
    _skriv_championlogg(logg, [
        {"modell": "SVR vanlig (T+6)",
         "timestamp": (datetime.now() - timedelta(days=2)).isoformat(),
         "sokvag": "svr_vanlig_(t+6)_Validation_R2_2026_10_21_10-00.joblib"},
        {"modell": f"Persistence Baseline{SUFFIX}",
         "timestamp": datetime.now().isoformat(),
         "sokvag": ""},
    ])

    modeller = tmp_path / "models"
    modeller.mkdir()
    joblib.dump("dummy", modeller / "svr_vanlig_(t+6)_Validation_R2_2026_10_21_10-00.joblib")
    # Forskjuten-linjens fil påverkar aldrig den vanliga championen —
    # loggen pekar aldrig på den.
    joblib.dump(
        "forskjuten-dummy",
        modeller / "forskjuten_svr_forskjuten_(t+6)_Validation_R2_2026_10_21_11-00.joblib",
    )

    modell, namn = hamta_champion_modell(modeller, logg)

    assert namn == "SVR vanlig (T+6)"   # baseline-raden är senast men utesluten
    assert modell == "dummy"


def _skriv_modell(modeller, namn):
    """Dummy-modellfil + metadatafil med gemensam namnstam."""
    joblib.dump(f"modell-{namn}", modeller / f"{namn}.joblib")
    (modeller / f"{namn}.json").write_text("{}", encoding="utf-8")


def test_stada_models_dir_behaller_refererade_och_fallback(tmp_path):
    """keep=2 per linje: aktuell champion + en fallback behålls, äldre
    generationer tas bort (.joblib + .json tillsammans). Forskjuten-
    linjens filer raderas aldrig av den vanliga linjens städning."""
    modeller = tmp_path / "models"
    modeller.mkdir()
    # Vanliga linjen: current=b, fallback=a, föråldrad=gammal
    _skriv_modell(modeller, "b")
    _skriv_modell(modeller, "a")
    _skriv_modell(modeller, "gammal")
    # Forskjutna linjen: en fil
    _skriv_modell(modeller, "forskjuten_b")
    # Fil som inte tillhör projektet lämnas orörd
    (modeller / "anteckningar.txt").write_text("oviktigt", encoding="utf-8")

    vanlig_logg = tmp_path / "champion_vanlig.csv"
    _skriv_championlogg(vanlig_logg, [
        {"modell": "SVR vanlig (T+6)", "timestamp": "2026-11-01T10:00:00",
         "sokvag": "gammal.joblib"},
        {"modell": "RF vanlig (T+6)", "timestamp": "2026-11-08T10:00:00",
         "sokvag": "a.joblib"},
        {"modell": "SVR vanlig (T+6)", "timestamp": "2026-11-14T10:00:00",
         "sokvag": "b.joblib"},
        # Baseline-vinst loggas utan sökväg och skyddar ingen fil
        {"modell": f"Persistence Baseline{SUFFIX}",
         "timestamp": "2026-11-15T10:00:00", "sokvag": ""},
    ])
    forskjuten_logg = tmp_path / "champion_forskjuten.csv"
    _skriv_championlogg(forskjuten_logg, [
        {"modell": "SVR forskjuten (T+6)", "timestamp": "2026-11-14T10:00:00",
         "sokvag": "forskjuten_b.joblib"},
    ])

    borttagna = stada_models_dir(
        modeller,
        [(vanlig_logg, ""), (forskjuten_logg, "forskjuten_")],
        keep=2,
    )

    assert sorted(borttagna) == ["gammal.joblib", "gammal.json"]
    for behallen in ("a", "b", "forskjuten_b"):
        assert (modeller / f"{behallen}.joblib").is_file()
        assert (modeller / f"{behallen}.json").is_file()
    assert (modeller / "anteckningar.txt").is_file()


def test_stada_models_dir_saknad_logg_raderar_ingenting(tmp_path):
    """Fail-safe: saknas den vanliga linjens logg vet vi inte vilka filer
    som är aktuella — prefixet '' matchar allt, så inget raderas."""
    modeller = tmp_path / "models"
    modeller.mkdir()
    _skriv_modell(modeller, "a")
    _skriv_modell(modeller, "forskjuten_b")

    borttagna = stada_models_dir(
        modeller,
        [(tmp_path / "champion_vanlig.csv", ""),
         (tmp_path / "champion_forskjuten.csv", "forskjuten_")],
        keep=2,
    )

    assert borttagna == []
    assert (modeller / "a.joblib").is_file()
    assert (modeller / "forskjuten_b.joblib").is_file()


def test_stada_models_dir_oleslbar_logg_raderar_ingenting(tmp_path):
    """Fail-safe: en korrupt logg ska aldrig leda till raderade filer."""
    modeller = tmp_path / "models"
    modeller.mkdir()
    _skriv_modell(modeller, "a")

    olelbar = tmp_path / "champion_vanlig.csv"
    olelbar.write_text("detta är inte csv{{", encoding="utf-8")

    borttagna = stada_models_dir(modeller, [(olelbar, "")], keep=2)

    assert borttagna == []
    assert (modeller / "a.joblib").is_file()


def _skriv_prediktionslogg(path, rader):
    pd.DataFrame(rader).to_csv(path, index=False)


def test_trim_prediktioner_log_rensar_gammal_champion_era(tmp_path):
    """Championbyte: raderna från den gamla championens era för linjen
    trimmas, andra linjen lämnas orörd och nya championen behålls."""
    logg = tmp_path / "prediktioner_live.csv"
    _skriv_prediktionslogg(logg, [
        {"linje": "vanlig", "modell": "Gammal vanlig (T+6)", "t_obs": "1"},
        {"linje": "forskjuten", "modell": "SVR forskjuten (T+6)", "t_obs": "2"},
        {"linje": "vanlig", "modell": "Gammal vanlig (T+6)", "t_obs": "3"},
        {"linje": "vanlig", "modell": "Ny vanlig (T+6)", "t_obs": "4"},
    ])

    antal = trim_prediktioner_log(
        logg, linje="vanlig", behall_modell="Ny vanlig (T+6)",
    )

    assert antal == 2
    trimmad = pd.read_csv(logg)
    # pandas läser t_obs som tal — raderna för gamla championen ("1", "3")
    # är borta, forskjuten-raden och nya championen ("2", "4") finns kvar.
    assert list(trimmad["t_obs"]) == [2, 4]
    # Rubrikraden och kolumnerna bevaras
    assert set(trimmad.columns) == {"linje", "modell", "t_obs"}


def test_trim_prediktioner_log_inget_att_trimma(tmp_path):
    """Redan ny champion: loggen lämnas orörd (0 borttagna)."""
    logg = tmp_path / "prediktioner_live.csv"
    _skriv_prediktionslogg(logg, [
        {"linje": "vanlig", "modell": "Ny vanlig (T+6)", "t_obs": "1"},
    ])

    assert trim_prediktioner_log(
        logg, linje="vanlig", behall_modell="Ny vanlig (T+6)",
    ) == 0
    assert len(pd.read_csv(logg)) == 1


def test_trim_prediktioner_log_saknad_fil(tmp_path):
    """Ingen logg = inget att trimma, inga krascher."""
    assert trim_prediktioner_log(
        tmp_path / "finns_inte.csv", linje="vanlig", behall_modell="X",
    ) == 0


# ----------------------------------------------------------
# Nisch-linjen (plan 01): trim med flera modeller, metadata-gräns,
# schema-migrering av append-loggar, städning med tre linjer
# ----------------------------------------------------------

def test_trim_prediktioner_log_behaller_flera_modeller(tmp_path):
    """Routing delar linjen vanlig på eror från championen OCH
    nisch-championen: behall_modell som lista behåller bådas rader."""
    logg = tmp_path / "prediktioner_live.csv"
    _skriv_prediktionslogg(logg, [
        {"linje": "vanlig", "modell": "SVR vanlig (T+6)", "t_obs": "1"},
        {"linje": "vanlig", "modell": "Gammal vanlig (T+6)", "t_obs": "2"},
        {"linje": "vanlig", "modell": "Gammal nisch (T+6)", "t_obs": "3"},
        {"linje": "vanlig", "modell": "SVR nisch (T+6)", "t_obs": "4"},
        {"linje": "forskjuten", "modell": "SVR forskjuten (T+6)", "t_obs": "5"},
    ])

    antal = trim_prediktioner_log(
        logg,
        linje="vanlig",
        behall_modell=["SVR vanlig (T+6)", "SVR nisch (T+6)"],
    )

    assert antal == 2
    trimmad = pd.read_csv(logg)
    kvar = trimmad[trimmad["linje"] == "vanlig"]["modell"].tolist()
    assert kvar == ["SVR vanlig (T+6)", "SVR nisch (T+6)"]
    # Forskjuten-linjen lämnas orörd
    assert (trimmad["linje"] == "forskjuten").sum() == 1


def _spara_champion_med_grans(tmp_path, namn, threshold=None):
    """Sparar en dummy-champion via save_champion_model och returnerar
    sökvägen till .joblib-filen."""
    rad = pd.Series({"Modell": namn, "Validation High PM2.5 MAE": 1.0})
    return save_champion_model(
        model="dummy",
        model_name=namn,
        best_row=rad,
        models_dir=tmp_path,
        random_state=42,
        metric="Validation High PM2.5 MAE",
        filprefix="nisch_",
        threshold=threshold,
        threshold_rule="pm25_p90_training" if threshold is not None else None,
    )


def test_save_champion_model_sparar_grans_i_metadata(tmp_path):
    """Threshold + threshold_rule sparas i .json (plan 01): enda källan
    till gränsen för routern och utvärderingen."""
    sokvag = _spara_champion_med_grans(tmp_path, "SVR nisch (T+6)", threshold=14.4)

    assert sokvag.name.startswith("nisch_")

    meta = las_champion_metadata(tmp_path, sokvag.name)
    assert meta["threshold"] == 14.4
    assert meta["threshold_rule"] == "pm25_p90_training"
    assert meta["variant"] == "nisch_"
    assert meta["selection_metric"] == "Validation High PM2.5 MAE"

    # Metadatafilen delar namnstam med modellfilen
    assert (tmp_path / (sokvag.stem + ".json")).is_file()


def test_save_champion_model_utan_grans_saknar_nyckel(tmp_path):
    """Ordinarie linjer skickar ingen gräns — nycklarna ska inte finnas
    heller (bakåtkompatibelt)."""
    rad = pd.Series({"Modell": "SVR", "Validation MAE": 1.5})
    sokvag = save_champion_model(
        model="dummy",
        model_name="SVR vanlig (T+6)",
        best_row=rad,
        models_dir=tmp_path,
        random_state=42,
        metric="Validation R²",
    )

    meta = las_champion_metadata(tmp_path, sokvag.name)
    assert "threshold" not in meta
    assert "threshold_rule" not in meta


def test_las_champion_metadata_fail_saker(tmp_path):
    """Saknad fil, oläslig JSON och främmande namnstam → {} (routern
    stängs av istället för att krascha)."""
    _spara_champion_med_grans(tmp_path, "SVR nisch (T+6)", threshold=14.4)

    # Namnstam som inte motsvarar någon .json-fil
    assert las_champion_metadata(tmp_path, "finns_inte.joblib") == {}
    assert las_champion_metadata(tmp_path, "nisch_annan_namnstam") == {}
    assert las_champion_metadata(tmp_path / "saknad_mapp", "x.joblib") == {}

    # Korrupt JSON → {}
    (tmp_path / "korrupt.json").write_text("detta ar inte json{{", encoding="utf-8")
    assert las_champion_metadata(tmp_path, "korrupt.joblib") == {}


def test_las_champion_metadata_laeser_via_namnstam(tmp_path):
    """Metadata hittas oavsett om filnamnet kommer med eller utan
    .joblib-ändelse."""
    sokvag = _spara_champion_med_grans(tmp_path, "SVR nisch (T+6)", threshold=14.4)

    med_ändelse = las_champion_metadata(tmp_path, sokvag.name)
    utan_ändelse = las_champion_metadata(tmp_path, sokvag.stem)

    assert med_ändelse["threshold"] == 14.4
    assert utan_ändelse == med_ändelse


def test_append_prediktioner_log_migrerar_nya_kolumner(tmp_path):
    """Logg med gammal rubrik + nya rader med extra kolumn (nisch_
    threshold, plan 01): filen skrivs om med unionen, gamla rader får
    tomma värden — inga rader hamnar ur fas med rubriken."""
    logg = tmp_path / "prediktioner_live.csv"
    _skriv_prediktionslogg(logg, [
        {"linje": "vanlig", "modell": "SVR vanlig (T+6)", "t_obs": "1"},
        {"linje": "forskjuten", "modell": "SVR forskjuten (T+6)", "t_obs": "2"},
    ])

    append_prediktioner_log(logg, pd.DataFrame([
        {
            "linje": "vanlig",
            "modell": "SVR nisch (T+6)",
            "t_obs": "3",
            "nisch_threshold": 14.4,
        },
    ]))

    after = pd.read_csv(logg)
    assert list(after.columns) == ["linje", "modell", "t_obs", "nisch_threshold"]
    assert len(after) == 3
    # Gamla rader saknar värden för den nya kolumnen
    assert after.iloc[0]["nisch_threshold"] != after.iloc[0]["nisch_threshold"]
    assert after.iloc[2]["nisch_threshold"] == 14.4
    # Raderna är fortfarande desamma
    assert after.iloc[0]["modell"] == "SVR vanlig (T+6)"
    assert after.iloc[2]["modell"] == "SVR nisch (T+6)"


def test_append_logg_samma_schema_appendar_rakt(tmp_path):
    """Efter migreringen appendar rader med SAMMA schema utan att
    rubrikraden dubbleras."""
    logg = tmp_path / "jamforelse_champion.csv"
    df1 = pd.DataFrame([
        {"timestamp": "2026-11-15T10:00:00", "Modell": "A", "MAE": 1.0, "rader": 4},
    ])
    append_jamfordelse_log(logg, df1)
    append_jamfordelse_log(logg, df1)

    after = pd.read_csv(logg)
    assert len(after) == 2
    assert "Modell" in after.columns
    assert after["Modell"].tolist() == ["A", "A"]


def test_stada_models_dir_skyddar_nisch_linjen(tmp_path):
    """Tre linjer: nisch-filerna behålls via nisch-loggens keep-lista
    och föråldrade nisch-generationer tas bort — vanliga linjens
    städning (prefix "" matchar allt) får aldrig plocka bort nisch- eller
    förskjuten-filer."""
    modeller = tmp_path / "models"
    modeller.mkdir()
    _skriv_modell(modeller, "vanlig_nuvarande")
    _skriv_modell(modeller, "forskjuten_nuvarande")
    _skriv_modell(modeller, "nisch_a")
    _skriv_modell(modeller, "nisch_b")
    _skriv_modell(modeller, "nisch_c")

    vanlig_logg = tmp_path / "champion_vanlig.csv"
    _skriv_championlogg(vanlig_logg, [
        {"modell": "SVR vanlig (T+6)", "timestamp": "2026-11-14T10:00:00",
         "sokvag": "vanlig_nuvarande.joblib"},
    ])
    forskjuten_logg = tmp_path / "champion_forskjuten.csv"
    _skriv_championlogg(forskjuten_logg, [
        {"modell": "SVR forskjuten (T+6)", "timestamp": "2026-11-14T10:00:00",
         "sokvag": "forskjuten_nuvarande.joblib"},
    ])
    nisch_logg = tmp_path / "champion_nisch.csv"
    _skriv_championlogg(nisch_logg, [
        {"modell": "SVR nisch (T+6)", "timestamp": "2026-11-01T10:00:00",
         "sokvag": "nisch_a.joblib"},
        {"modell": "RF nisch (T+6)", "timestamp": "2026-11-08T10:00:00",
         "sokvag": "nisch_b.joblib"},
        {"modell": "SVR nisch (T+6)", "timestamp": "2026-11-15T10:00:00",
         "sokvag": "nisch_c.joblib"},
    ])

    borttagna = stada_models_dir(
        modeller,
        [
            (vanlig_logg, ""),
            (forskjuten_logg, "forskjuten_"),
            (nisch_logg, "nisch_"),
        ],
        keep=2,
    )

    # keep=2: aktuell + fallback per linje — bara nisch_a är föråldrad
    assert sorted(borttagna) == ["nisch_a.joblib", "nisch_a.json"]
    for behallen in ("vanlig_nuvarande", "forskjuten_nuvarande", "nisch_b", "nisch_c"):
        assert (modeller / f"{behallen}.joblib").is_file()
        assert (modeller / f"{behallen}.json").is_file()


# ----------------------------------------------------------
# Champion-loggens kompositkolumner (plan 02)
# ----------------------------------------------------------

def test_append_champion_log_migrerar_kompositkolumner(tmp_path):
    """Logg med gammal rubrik (pre-plan-02) + ny rad med kompositens
    delvärden: filen skrivs om med unionen, gamla rader får tomma
    fält för de nya kolumnerna."""
    logg = tmp_path / "champion_vanlig.csv"
    _skriv_championlogg(logg, [
        {"modell": "SVR vanlig (T+6)", "timestamp": "2026-11-01T10:00:00",
         "metric": "Test R²", "score": 0.60, "sokvag": "svr_gammal.joblib"},
    ])

    append_champion_log(
        log_path=logg,
        model_name="SVR vanlig (T+6)",
        metric="Composite",
        score=2.4100,
        sokvag=tmp_path / "svr_validation_composite_2026.joblib",
        extra={
            "score_mae": 2.55,
            "score_high_mae": 2.90,
            "score_high_bias": -1.30,
        },
    )

    logg_df = pd.read_csv(logg)
    for kolumn in ("score_mae", "score_high_mae", "score_high_bias"):
        assert kolumn in logg_df.columns

    # Gamla raden: tomma delvärden. Nya raden: ifyllda.
    gammal = logg_df.iloc[0]
    ny = logg_df.iloc[1]

    assert pd.isna(gammal["score_mae"])
    assert gammal["metric"] == "Test R²"

    assert ny["metric"] == "Composite"
    assert ny["score"] == 2.41
    assert ny["score_mae"] == 2.55
    assert ny["score_high_mae"] == 2.90
    assert ny["score_high_bias"] == -1.30


def test_append_champion_log_utan_extra_oforandrad_rubrik(tmp_path):
    """Utan extra-fält (t.ex. nisch-linjen) skrivs raden som vanligt
    append med oförändrad rubrik."""
    logg = tmp_path / "champion_nisch.csv"

    append_champion_log(
        log_path=logg,
        model_name="SVR nisch (T+6)",
        metric="Validation High PM2.5 MAE",
        score=2.90,
        sokvag=tmp_path / "nisch_svr.joblib",
    )

    logg_df = pd.read_csv(logg)
    assert list(logg_df.columns) == [
        "modell", "timestamp", "metric", "score", "sokvag",
    ]
    assert logg_df.iloc[0]["metric"] == "Validation High PM2.5 MAE"


# ----------------------------------------------------------
# Plan 04: datakontrollens mönsterkolumner, bortfallsanalys-loggen
# och skipp-loggen
# ----------------------------------------------------------

def _kolumnstatistik():
    """Mönsterstatistik-fixtur i sammanstall_bortfalls format."""
    return pd.DataFrame([
        {
            "kolumn": "pm25", "saknade": 17,
            "langsta_gap_timmar": 17, "antal_gap_over_horisont": 1,
        },
        {
            "kolumn": "Lufttemperatur", "saknade": 0,
            "langsta_gap_timmar": 0, "antal_gap_over_horisont": 0,
        },
    ])


def test_append_datakontroll_log_monsterkolumner(tmp_path):
    """Med kolumnstatistik (plan 04) får varje kolumn två nya kolumner —
    även de rena (0-gap) så att loggen får samma kolumner varje körning."""
    logg = tmp_path / "datakontroll_6247.csv"

    append_datakontroll_log(
        logg, "train_processed.csv", 100, 100, 90,
        pd.Series({"pm25": 17, "Lufttemperatur": 0}),
        kolumnstatistik=_kolumnstatistik(),
    )

    df = pd.read_csv(logg)
    assert df["langsta_gap_pm25_h"].iloc[0] == 17
    assert df["antal_gap_over_horisont_pm25"].iloc[0] == 1
    assert df["langsta_gap_Lufttemperatur_h"].iloc[0] == 0
    assert df["antal_gap_over_horisont_Lufttemperatur"].iloc[0] == 0
    assert df["saknade_pm25"].iloc[0] == 17


def test_append_datakontroll_log_migrerar_gammal_rubrikrad(tmp_path):
    """Befintlig datakontroll (före plan 04). Nya mönsterkolumner
    migreras in i rubrikraden — gamla rader får tomma fält, nya rader
    skrivs korrekt (samma mekanism som champion-loggen, plan 02)."""
    logg = tmp_path / "datakontroll_6247.csv"
    pd.DataFrame([{
        "timestamp": "2026-11-01T12:00:00", "dataset": "train.csv",
        "rader_merged": 10, "rader_validerad": 10, "rader_kvar": 9,
        "borttagna": 1, "andel_borttagna": 0.1,
    }]).to_csv(logg, index=False)

    append_datakontroll_log(
        logg, "train_processed.csv", 100, 100, 99,
        pd.Series({"pm25": 1}),
        kolumnstatistik=_kolumnstatistik(),
    )

    df = pd.read_csv(logg)
    assert len(df) == 2
    # Gamla raden: tom för de nya kolumnerna
    assert pd.isna(df["langsta_gap_pm25_h"].iloc[0])
    # Nya raden: skriven korrekt
    assert df["langsta_gap_pm25_h"].iloc[1] == 17


def test_append_bortfallsanalys_log_skriver_en_rad_per_gap(tmp_path):
    logg = tmp_path / "bortfallsanalys_6247.csv"
    start = pd.Timestamp("2026-05-01 00:00")
    gaprader = pd.DataFrame([
        {
            "kolumn": "Nederbördsmängd", "start": start,
            "slut": start + pd.Timedelta(hours=11),
            "langd_timmar": 12, "manad": 5,
        },
        {
            "kolumn": "pm25", "start": start + pd.Timedelta(hours=24),
            "slut": start + pd.Timedelta(hours=40),
            "langd_timmar": 17, "manad": 5,
        },
    ])

    append_bortfallsanalys_log(logg, "train_processed.csv", gaprader)

    df = pd.read_csv(logg)
    assert len(df) == 2
    assert (
        list(df.columns)
        == ["timestamp", "dataset", "kolumn", "start", "slut", "langd_timmar", "manad"]
    )
    assert (df["dataset"] == "train_processed.csv").all()
    assert (df["manad"] == 5).all()


def test_append_bortfallsanalys_log_utan_gap_skriver_inget(tmp_path):
    """Ett rent dataset ska inte generera brus — datakontroll-raden
    bevisar att kontrollen kördes (langsta_gap-kolumnerna = 0)."""
    logg = tmp_path / "bortfallsanalys_6247.csv"

    append_bortfallsanalys_log(logg, "train_processed.csv", pd.DataFrame())
    append_bortfallsanalys_log(logg, "val_processed.csv", None)

    assert not logg.exists()


def test_append_skippad_log_en_rad_per_overhoppning(tmp_path):
    """Rubrikraden skapas vid första raden; varje överhoppning blir exakt
    en rad; valfria fält lämnas TOMMA (inte 'nan')."""
    logg = tmp_path / "live_skippad_6247.csv"

    append_skippad_log(logg, orsak="ingen_ankarpunkt")
    append_skippad_log(
        logg, orsak="for_gammal_ankarpunkt",
        t_obs=pd.Timestamp("2026-11-15 10:00", tz="Europe/Stockholm"),
        pm25_alder_h=7.499,
    )

    df = pd.read_csv(logg)
    assert len(df) == 2
    assert list(df.columns) == [
        "timestamp", "t_obs", "orsak", "linje", "alinje", "pm25_alder_h",
    ]

    rad1, rad2 = df.iloc[0], df.iloc[1]
    assert rad1["orsak"] == "ingen_ankarpunkt"
    assert rad1["alinje"] == "hela körningen"
    # T_obs hann inte bestämmas -> tomt fält, inte 'nan'
    assert pd.isna(rad1["t_obs"]) and pd.isna(rad1["pm25_alder_h"])

    assert rad2["orsak"] == "for_gammal_ankarpunkt"
    assert rad2["pm25_alder_h"] == 7.5
    assert "2026-11-15" in rad2["t_obs"]


def test_append_skippad_log_linjeniva(tmp_path):
    """En ENKEL linje som hoppas över loggas med linje + alinje='linje'."""
    logg = tmp_path / "live_skippad_6247.csv"

    append_skippad_log(
        logg, orsak="prognos_saknas_maltid",
        t_obs=pd.Timestamp("2026-11-15 10:00", tz="Europe/Stockholm"),
        linje="forskjuten", alinje="linje", pm25_alder_h=1.2,
    )
    append_skippad_log(
        logg, orsak="champion_saknas:vanlig",
        t_obs=pd.Timestamp("2026-11-15 10:00", tz="Europe/Stockholm"),
        linje="vanlig", alinje="linje",
    )

    df = pd.read_csv(logg)
    assert df["alinje"].tolist() == ["linje", "linje"]
    assert df["linje"].tolist() == ["forskjuten", "vanlig"]
    assert df["orsak"].tolist() == [
        "prognos_saknas_maltid", "champion_saknas:vanlig",
    ]

