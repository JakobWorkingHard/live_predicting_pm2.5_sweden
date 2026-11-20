"""Tester för src/model_pipeline.py: nisch-linjen (plan 01, beslut
2026-11-15) — adoptionsreglerna mot den ordinarie championen och en
eventuell tidigare nisch-champion, persistenta-champion-sparningen,
gränsen ur metadata samt färskhetsgenvägen."""

from datetime import datetime, timedelta
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest

import src.model_pipeline as mp
from src.io import las_champion_metadata
from src.model_pipeline import run_nisch_linje
from src.train_models import PersistenceModel, berakna_high_pm_threshold


class KonstantModell:
    """Dummy som predikterar samma värde oavsett input."""

    def __init__(self, varde):
        self.varde = varde

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.full(len(X), self.varde)


def _skriv_championlogg(path, rader):
    pd.DataFrame(rader).to_csv(path, index=False)


def _skriv_prediktionslogg(path, rader):
    pd.DataFrame(rader).to_csv(path, index=False)


def _spara_ordinarie_champion(models_dir, champion_log, modell, namn):
    """Sparar en ordinarie champion manuellt (fil + loggrad med sokvag)."""
    fil = models_dir / "ordnarie_champion_test.joblib"
    joblib.dump(modell, fil)
    _skriv_championlogg(champion_log, [
        {"modell": namn, "timestamp": datetime.now().isoformat(),
         "sokvag": fil.name},
    ])


def _nisch_data(data_dir, target_funktion=None):
    """
    Träning: pm25 1..1010 (p90 ~ 909 -> drygt 100 filtrerade rader).
    Validering: 950..1000 (över gränsen) plus 5, 6, 7 (under).
    target_funktion ersätter standard-y = pm25 (persistence perfekt).
    """
    if target_funktion is None:
        target_funktion = lambda v: v  # noqa: E731

    train = pd.DataFrame({"pm25": list(range(1, 1011))})
    train["target"] = train["pm25"].map(target_funktion)

    val = pd.DataFrame({"pm25": list(range(950, 1001)) + [5, 6, 7]})
    val["target"] = val["pm25"].map(target_funktion)

    train_path = data_dir / "train.csv"
    val_path = data_dir / "val.csv"
    train.to_csv(train_path, index=False)
    val.to_csv(val_path, index=False)

    return train_path, val_path


def _cfg(tmp_path):
    return SimpleNamespace(
        features=["pm25"],
        target_column="target",
        random_state=42,
        models_dir=tmp_path / "models",
        champion_log=tmp_path / "champion_vanlig.csv",
        nisch_log=tmp_path / "champion_nisch.csv",
        prediktioner_log=tmp_path / "prediktioner_live.csv",
        high_pm_threshold_rule="pm25_p90_training",
    )


def _forvantad_grans(train_path):
    train = pd.read_csv(train_path)
    return berakna_high_pm_threshold(train[["pm25"]])


@pytest.fixture
def uppsatt(tmp_path, monkeypatch):
    """
    Standardscenario för adoptions testerna: y = pm25 (persistence
    perfekt), registret är monkeypatchat till en dålig konstantmodell
    (så att Persistence Baseline vinner kandidaturvalet deterministiskt)
    och den ordinarie championen är en dålig konstantmodell.
    """
    cfg = _cfg(tmp_path)
    cfg.models_dir.mkdir(parents=True)

    train_path, val_path = _nisch_data(tmp_path)

    # Ordinarie champion: dålig på höga rader (High MAE ~ 750)
    _spara_ordinarie_champion(
        cfg.models_dir, cfg.champion_log,
        KonstantModell(300.0), "SVR vanlig (T+6)",
    )

    # Prediktionslogg med gamla eror: ordinarie (behålls i trim),
    # gammal vanlig (bort), gammal nisch (bort), forskjuten (orörd).
    # Rad 5 har den KOMMANDE nisch-championens namn — en era med samma
    # modellnamn ska överleva trimmen vid nästa adoption.
    _skriv_prediktionslogg(cfg.prediktioner_log, [
        {"linje": "vanlig", "modell": "SVR vanlig (T+6)", "t_obs": "1"},
        {"linje": "vanlig", "modell": "Gammal vanlig (T+6)", "t_obs": "2"},
        {"linje": "vanlig", "modell": "Gammal nisch (T+6)", "t_obs": "3"},
        {"linje": "forskjuten", "modell": "SVR forskjuten (T+6)", "t_obs": "4"},
        {"linje": "vanlig", "modell": "Persistence Baseline nisch (T+6)",
         "t_obs": "5"},
    ])

    monkeypatch.setattr(
        mp,
        "build_model_registry",
        lambda random_state: {"Linear Regression": KonstantModell(300.0)},
    )

    return cfg, train_path, val_path


def test_adoption_persistence_sparas_och_loggas(uppsatt):
    """Vinner baslinjen nisch-turneringen antas den som
    PersistenceModel: sparad med gräns i metadata, champion-loggad och
    prediktionsloggen trimmad korrekt (champion + nisch behålls)."""
    cfg, train_path, val_path = uppsatt

    modell, namn, threshold = run_nisch_linje(
        cfg, train_path, val_path, antal_dagar=7,
        name_suffix=" nisch (T+6)",
    )

    forvantad = _forvantad_grans(train_path)

    # Vinnare och gräns
    assert namn == "Persistence Baseline nisch (T+6)"
    assert threshold == pytest.approx(forvantad)
    assert isinstance(modell, PersistenceModel)

    # Modellfil + metadata med gränsen
    nisch_filer = list(cfg.models_dir.glob("nisch_*.joblib"))
    assert len(nisch_filer) == 1
    meta = las_champion_metadata(cfg.models_dir, nisch_filer[0].name)
    assert meta["threshold"] == pytest.approx(forvantad)
    assert meta["threshold_rule"] == cfg.high_pm_threshold_rule

    # Champion-loggen: en rad med sokvag och rätt mått
    logg = pd.read_csv(cfg.nisch_log)
    assert len(logg) == 1
    assert logg.iloc[0]["modell"] == "Persistence Baseline nisch (T+6)"
    assert logg.iloc[0]["sokvag"] == nisch_filer[0].name
    assert logg.iloc[0]["metric"] == "Validation High PM2.5 MAE"

    # Era-trim: ordinarie championens rad behålls, gamla eror borta,
    # forskjuten-linjen orörd
    trimmad = pd.read_csv(cfg.prediktioner_log)
    vanliga = trimmad[trimmad["linje"] == "vanlig"]["modell"].tolist()
    assert vanliga == ["SVR vanlig (T+6)", "Persistence Baseline nisch (T+6)"]
    assert (trimmad["linje"] == "forskjuten").sum() == 1

    # Modellen "predikterar" nuvärdet (persistence)
    X = pd.DataFrame({"pm25": [10.0, 20.0]})
    assert list(modell.predict(X)) == [10.0, 20.0]


def test_tie_mot_ordinarie_champion_ingen_adoption(uppsatt):
    """Lika med den ordinarie championen (här: persistence båda) =>
    det strikta kravet missas => ingen nisch-champion sparas, ingen
    logg och prediktionsloggen lämnas orörd."""
    cfg, train_path, val_path = uppsatt

    # Ordinarie champion = PersistenceModel => identiska prediktioner
    _spara_ordinarie_champion(
        cfg.models_dir, cfg.champion_log,
        PersistenceModel(), "SVR vanlig (T+6)",
    )

    resultat = run_nisch_linje(
        cfg, train_path, val_path, antal_dagar=7,
        name_suffix=" nisch (T+6)",
    )

    assert resultat == (None, None, None)
    assert not cfg.nisch_log.is_file()
    assert list(cfg.models_dir.glob("nisch_*.joblib")) == []

    # Ingen trim — loggen orörd
    trimmad = pd.read_csv(cfg.prediktioner_log)
    assert len(trimmad) == 5


def test_samre_an_ordinarie_champion_ingen_adoption(uppsatt):
    """Kandidaten är sämre än den ordinarie championen (bra champion,
    persistence missar i datan) => avvisad duel: inget sparas, ingen
    logg skapas."""
    cfg, *_ = uppsatt

    # Ordinarie champion = PersistenceModel (High MAE 0). Data där
    # persistence missar med 5 => kandidatens fel 5 slår inte 0.
    train_path, val_path = _nisch_data(
        cfg.models_dir.parent, target_funktion=lambda v: v + 5.0,
    )
    _spara_ordinarie_champion(
        cfg.models_dir, cfg.champion_log,
        PersistenceModel(), "SVR vanlig (T+6)",
    )

    resultat = run_nisch_linje(
        cfg, train_path, val_path, antal_dagar=7,
        name_suffix=" nisch (T+6)",
    )

    assert resultat == (None, None, None)
    assert not cfg.nisch_log.is_file()


def test_tidigare_nisch_champion_behalls_vid_tie(uppsatt):
    """Gate A passeras men kandidaten slår inte den befintliga
    nisch-championen strikt => den gamla behålls, ingen ny rad i
    champion_nisch.csv och ingen extra modellfil."""
    cfg, train_path, val_path = uppsatt

    # Kör 1: adoption (samme scenario som adoptionstestet)
    forsta = run_nisch_linje(
        cfg, train_path, val_path, antal_dagar=7,
        name_suffix=" nisch (T+6)",
    )
    assert forsta[1] == "Persistence Baseline nisch (T+6)"

    # Kör 2: antal_dagar=0 tvingar fram duellen igen. Kandidaten blir
    # densamma (persistence, monkeypatchat register) => tie mot den
    # gamla nisch-championen => den gamla behålls.
    andra = run_nisch_linje(
        cfg, train_path, val_path, antal_dagar=0,
        name_suffix=" nisch (T+6)",
    )

    assert andra == (None, None, None)

    logg = pd.read_csv(cfg.nisch_log)
    assert len(logg) == 1  # ingen ny rad
    assert logg.iloc[0]["modell"] == "Persistence Baseline nisch (T+6)"
    assert len(list(cfg.models_dir.glob("nisch_*.joblib"))) == 1


def test_ny_kandidat_ersatter_gammal_nisch_champion(uppsatt, monkeypatch):
    """Gate A + B: en bättre kandidat ersätter en gammal nisch-champion
    med högre fel — ny rad i champion-loggen, gammal era trimmas bort."""
    cfg, *_ = uppsatt
    tmp_dir = cfg.models_dir.parent

    # Data där persistence missar med exakt 5 => persistence vinner
    # kandidaturvalet (monkeypatchat register: konstant 300) och antas
    # i kör 1.
    train_path, val_path = _nisch_data(
        tmp_dir, target_funktion=lambda v: v + 5.0,
    )

    forsta = run_nisch_linje(
        cfg, train_path, val_path, antal_dagar=7,
        name_suffix=" nisch (T+6)",
    )
    assert forsta[1] == "Persistence Baseline nisch (T+6)"

    # Kör 2: RIKTIGT register — modellfamiljerna lär sig y = pm25 + 5
    # nästan exakt (High MAE ~ 0) och slår därmed den gamla
    # nisch-championens fel 5,0 strikt.
    monkeypatch.undo()
    modell, namn, threshold = run_nisch_linje(
        cfg, train_path, val_path, antal_dagar=0,
        name_suffix=" nisch (T+6)",
    )

    assert namn.endswith(" nisch (T+6)")
    assert namn != "Persistence Baseline nisch (T+6)"
    assert not isinstance(modell, PersistenceModel)
    assert threshold == pytest.approx(_forvantad_grans(train_path))

    # Två loggade nisch-generationer; den senaste är den nya
    logg = pd.read_csv(cfg.nisch_log)
    assert len(logg) == 2
    assert logg.iloc[-1]["modell"] == namn

    # Era-trim: den gamla nisch-erans rader har tagits bort; rader
    # för den nya championen finns ännu inte i loggen (de skapas vid
    # nästa live-prediktion) — det är bara trimskyddet som gäller.
    trimmad = pd.read_csv(cfg.prediktioner_log)
    vanliga = trimmad[trimmad["linje"] == "vanlig"]["modell"].tolist()
    assert "Gammal nisch (T+6)" not in vanliga
    assert "Gammal vanlig (T+6)" not in vanliga
    assert "SVR vanlig (T+6)" in vanliga


def test_ingen_ordinarie_champion_inget_adoption(uppsatt):
    """Finns ingen ordinarie champion-modell kan duellen inte köras =>
    ingen adoption den körningen (fail-säkert, utan krasch)."""
    cfg, train_path, val_path = uppsatt
    cfg.champion_log.unlink()  # ingen ordinarie champion alls

    resultat = run_nisch_linje(
        cfg, train_path, val_path, antal_dagar=7,
        name_suffix=" nisch (T+6)",
    )

    assert resultat == (None, None, None)
    assert not cfg.nisch_log.is_file()


def test_for_fa_nisch_rader_hoppas_over(tmp_path):
    """Under minimiantalet filtrerade träningsrader (~100) hoppas
    nisch-linjen över — inget sparas, inget loggas."""
    import src.model_pipeline as mp_modul

    cfg = _cfg(tmp_path)
    cfg.models_dir.mkdir(parents=True)

    # 30 rader: p90 ~ 28 => 3 filtrerade rader < _NISCH_MIN_TRADER
    train = pd.DataFrame({"pm25": list(range(1, 31))})
    train["target"] = train["pm25"]
    val = pd.DataFrame({"pm25": [29, 30], "target": [29, 30]})

    train_path = tmp_path / "train.csv"
    val_path = tmp_path / "val.csv"
    train.to_csv(train_path, index=False)
    val.to_csv(val_path, index=False)

    _spara_ordinarie_champion(
        cfg.models_dir, cfg.champion_log,
        KonstantModell(300.0), "SVR vanlig (T+6)",
    )

    resultat = run_nisch_linje(
        cfg, train_path, val_path, antal_dagar=7,
        name_suffix=" nisch (T+6)",
    )

    assert resultat == (None, None, None)
    assert not cfg.nisch_log.is_file()
    assert mp_modul._NISCH_MIN_TRADER == 100


def test_farsk_nisch_champion_laddas_utan_omtraning(uppsatt, monkeypatch):
    """Är nisch-championen färsk returneras den inlästa modellen med
    gränsen ur metadata — turneringen körs inte igen."""
    cfg, train_path, val_path = uppsatt

    modell1, namn1, grans1 = run_nisch_linje(
        cfg, train_path, val_path, antal_dagar=7,
        name_suffix=" nisch (T+6)",
    )

    # Något register som KRASCHAR om det används: färskhetsgenvägen
    # får aldrig träna något.
    def krash(random_state):
        raise AssertionError(
            "Turneringen fick inte köras — championen skulle vara färsk."
        )

    monkeypatch.setattr(mp, "build_model_registry", krash)

    modell2, namn2, grans2 = run_nisch_linje(
        cfg, train_path, val_path, antal_dagar=7,
        name_suffix=" nisch (T+6)",
    )

    assert namn2 == namn1
    assert grans2 == pytest.approx(grans1)
    assert hasattr(modell2, "predict")
    assert len(pd.read_csv(cfg.nisch_log)) == 1  # ingen ny loggrad


# ----------------------------------------------------------
# Ordinarie modellinje (plan 02): kompositval + valideringsduell
# ----------------------------------------------------------

def test_run_model_line_vinnare_loggas_med_kompositdelvarden(tmp_path, monkeypatch):
    """Plan 02, integration: en linje utan tidigare champion korrar
    hela flödet — turnering, kompositval, automatduell, sparad champion
    och en championlogg-rad med metric='Composite' samt delvärdena."""
    from src.model_pipeline import run_model_line
    from sklearn.linear_model import LinearRegression

    models_dir = tmp_path / "models"
    models_dir.mkdir()

    train = pd.DataFrame({"pm25": list(range(1, 61))})
    train["target"] = 2 * train["pm25"]          # LinReg kan lära sig exakt
    val = pd.DataFrame({"pm25": list(range(61, 81))})
    val["target"] = 2 * val["pm25"]

    train_path = tmp_path / "train.csv"
    val_path = tmp_path / "val.csv"
    train.to_csv(train_path, index=False)
    val.to_csv(val_path, index=False)

    cfg = SimpleNamespace(
        features=["pm25"],
        target_column="target",
        random_state=42,
        candidate_selection_metric="Validation Composite",
        champion_selection_metric="Validation Composite",
        composite_vikter={
            "w_mae": 0.5, "w_high_mae": 0.35, "w_high_bias": 0.15,
            "min_antal_high": 3,
        },
        models_dir=models_dir,
        prediktioner_log=tmp_path / "prediktioner_live.csv",
        hparam_aktiverad=False,
        hparam_grids={},
    )

    # Snabbt "register": enbart Linear Regression.
    monkeypatch.setattr(
        mp, "build_model_registry",
        lambda random_state, bast_params=None: {"Linear Regression": LinearRegression()},
    )

    modell, namn = run_model_line(
        cfg,
        line_label="vanlig",
        train_path=train_path,
        val_path=val_path,
        log_path=tmp_path / "champion_vanlig.csv",
        antal_dagar=7,
        name_suffix=" vanlig (T+6)",
    )

    # LinReg vinner över persistence (fel ~pm25), champion sparas
    assert namn == "Linear Regression vanlig (T+6)"

    logg = pd.read_csv(tmp_path / "champion_vanlig.csv")
    assert len(logg) == 1
    rad = logg.iloc[0]

    # Plan 02: metric-kolumnen är kompositen, med delvärden på raden
    assert rad["metric"] == "Composite"
    for kolumn in ("score_mae", "score_high_mae", "score_high_bias"):
        assert kolumn in logg.columns
        assert not pd.isna(rad[kolumn])

    # Perfekt modell => kompositen ~0 och slår baslinjens
    assert rad["score"] == pytest.approx(0.0, abs=1e-6)

    # Modellen + metadata (vikter och utvärderingsgräns) sparade
    joblib_filer = list(models_dir.glob("*.joblib"))
    assert len(joblib_filer) == 1
    meta = las_champion_metadata(models_dir, joblib_filer[0].name)
    assert meta["selection_metric"] == "Validation Composite"
    assert meta["composite_vikter"]["w_mae"] == 0.5
    assert "utvarderings_grans" in meta

    # Ingen testfil behövs — och skickas inte längre in heller:
    # signaturen tog bort test_path (plan 02). Rör inte flödet.
    assert modell is not None


def test_run_model_line_gammal_champion_behalls_vid_tie(tmp_path, monkeypatch):
    """Plan 02: kandidaten som inte är STRIKT bättre förlorar duellen —
    gamla championen loggas igen med dagens komposit (färskhetsklockan
    skal inte tvinga fram omträning varje körning).

    Tie konstrueras deterministiskt: den gamla championen är en LinReg
    tränad på exakt samma data som kandidaten (LinReg-fit är
    deterministisk) => identiska prediktioner => identisk komposit."""
    from src.model_pipeline import run_model_line
    from sklearn.linear_model import LinearRegression

    models_dir = tmp_path / "models"
    models_dir.mkdir()

    train = pd.DataFrame({"pm25": list(range(1, 61))})
    train["target"] = 2 * train["pm25"]      # LinReg kan lära sig exakt
    val = pd.DataFrame({"pm25": list(range(61, 81))})
    val["target"] = 2 * val["pm25"]

    train_path = tmp_path / "train.csv"
    val_path = tmp_path / "val.csv"
    train.to_csv(train_path, index=False)
    val.to_csv(val_path, index=False)

    logg_path = tmp_path / "champion_vanlig.csv"

    # Ordinarie champion: LinReg tränad på samma data — kandidaten blir
    # identisk => composite-tie => gamla championen behålls.
    gammal = LinearRegression().fit(train[["pm25"]], train["target"])
    joblib.dump(gammal, models_dir / "gammal_champion.joblib")
    # 8 dagar gammal timestamp => inte färsk (gränsen är 7) => duellen körs.
    _skriv_championlogg(logg_path, [
        {"modell": "Linear Regression vanlig (T+6)",
         "timestamp": (datetime.now() - timedelta(days=8)).isoformat(),
         "metric": "Composite",
         "score": 0.0,
         "sokvag": "gammal_champion.joblib"},
    ])

    cfg = SimpleNamespace(
        features=["pm25"],
        target_column="target",
        random_state=42,
        candidate_selection_metric="Validation Composite",
        champion_selection_metric="Validation Composite",
        composite_vikter={
            "w_mae": 0.5, "w_high_mae": 0.35, "w_high_bias": 0.15,
            "min_antal_high": 3,
        },
        models_dir=models_dir,
        prediktioner_log=tmp_path / "prediktioner_live.csv",
        hparam_aktiverad=False,
        hparam_grids={},
    )

    monkeypatch.setattr(
        mp, "build_model_registry",
        lambda random_state, bast_params=None: {"Linear Regression": LinearRegression()},
    )

    modell, namn = run_model_line(
        cfg,
        line_label="vanlig",
        train_path=train_path,
        val_path=val_path,
        log_path=logg_path,
        antal_dagar=7,
        name_suffix=" vanlig (T+6)",
    )

    # Gamla championen behålls och returneras
    assert namn == "Linear Regression vanlig (T+6)"
    assert modell is not None

    logg = pd.read_csv(logg_path)
    assert len(logg) == 2  # gamla championen loggades igen, ingen ny modell
    assert logg.iloc[-1]["modell"] == "Linear Regression vanlig (T+6)"
    assert logg.iloc[-1]["sokvag"] == "gammal_champion.joblib"
    assert logg.iloc[-1]["metric"] == "Composite"
    for kolumn in ("score_mae", "score_high_mae", "score_high_bias"):
        assert kolumn in logg.columns
        assert not pd.isna(logg.iloc[-1][kolumn])

    # Ingen NY modellfil sparades för duellförloraren
    assert len(list(models_dir.glob("*.joblib"))) == 1
