import json

import numpy as np
import pandas as pd
import pytest

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit

import src.train_models as tm
from src.train_models import (
    train_and_evaluate_models,
    persistence_baseline,
    build_model_registry,
    high_pm_score,
    high_pm_scorer,
    sok_basta_params,
    sok_parametrar_for_registry,
    berakna_high_pm_threshold,
    filtrera_high_pm,
    PersistenceModel,
)


@pytest.fixture
def sample_data():
    """Skapar enkla testdata för träning och validering."""

    X_train = pd.DataFrame({
        "pm25": [10, 15, 20, 25, 30, 40, 50, 60, 70, 80],
    })

    y_train = pd.Series(
        [12, 17, 22, 27, 32, 42, 52, 62, 72, 82]
    )

    X_val = pd.DataFrame({
        "pm25": [15, 30, 50, 75],
    })

    y_val = pd.Series([17, 32, 52, 77])

    return X_train, y_train, X_val, y_val


# ----------------------------------------------------------
# 1. Test av modellträning och utvärdering
# ----------------------------------------------------------

def test_train_and_evaluate_models(sample_data):

    X_train, y_train, X_val, y_val = sample_data

    models = {
        "Linear Regression": LinearRegression()
    }

    results_df, trained_models = train_and_evaluate_models(
        models, X_train, y_train, X_val, y_val
    )

    # Kontrollera att modellen tränats
    assert "Linear Regression" in trained_models
    assert len(results_df) == 1

    # Kontrollera att alla mått finns
    expected_columns = [
        "Train MAE",
        "Train RMSE",
        "Train R²",
        "Train Bias",
        "Train High PM2.5 MAE",
        "Validation MAE",
        "Validation RMSE",
        "Validation R²",
        "Validation Bias",
        "Validation High PM2.5 MAE",
        "Validation High PM2.5 Bias",
        "Validation High PM2.5 Count",
    ]

    for col in expected_columns:
        assert col in results_df.columns

    # Kontrollera gränsvärdet indirekt:
    # 90:e percentilen av y_train = 73.0
    # Endast y_val = 77 ligger över gränsen.
    assert results_df.iloc[0]["Validation High PM2.5 Count"] == 1

    # Modellen bör lära sig sambandet y = pm25 + 2
    assert results_df.iloc[0]["Validation MAE"] == pytest.approx(0)


# ----------------------------------------------------------
# 2. Test av persistence baseline
# ----------------------------------------------------------

def test_persistence_baseline(sample_data):

    X_train, y_train, X_val, y_val = sample_data

    train_metrics, val_metrics = persistence_baseline(
        X_train, y_train, X_val, y_val
    )

    # Kontrollera att samtliga mått finns
    expected_metrics = [
        "MAE",
        "RMSE",
        "R2",
        "Bias",
        "High PM2.5 MAE",
        "High PM2.5 Bias",
        "High PM2.5 Count",
    ]

    for metric in expected_metrics:
        assert metric in train_metrics
        assert metric in val_metrics

    # Baseline predikterar pm25 utan förändring,
    # medan faktiska värden är pm25 + 2.
    assert val_metrics["MAE"] == pytest.approx(2.0)
    assert val_metrics["Bias"] == pytest.approx(-2.0)

    # En hög observation i valideringsdata
    assert val_metrics["High PM2.5 Count"] == 1
    assert val_metrics["High PM2.5 MAE"] == pytest.approx(2.0)


# ----------------------------------------------------------
# 3. Hyperparametersökning (plan 05)
# ----------------------------------------------------------

@pytest.fixture
def ren_sok_cache():
    """Nollställer sök-cachen (mellan- och fil-cachen) före/efter test."""
    tm._SOK_MEMO.clear()
    tm._HPARAM_CACHE_SKRIVEN = False
    yield
    tm._SOK_MEMO.clear()
    tm._HPARAM_CACHE_SKRIVEN = False


def _sokdata(rader=80, fro=42):
    """Deterministisk syntetisk tidsserie för söktesten."""
    slump = np.random.default_rng(fro)
    X = pd.DataFrame({"pm25": slump.normal(loc=10, scale=2, size=rader)})
    y = pd.Series(2 * X["pm25"] + 1 + slump.normal(scale=0.5, size=rader))
    return X, y


def test_time_series_split_gap_forebygger_maldackage():
    """gap=horisont: träningsfadens mål (T+horisont) hamnar aldrig i valdfadet."""
    cv = TimeSeriesSplit(n_splits=3, gap=6)
    for train_idx, val_idx in cv.split(np.arange(60)):
        assert train_idx.max() + 6 < val_idx.min()


def test_high_pm_scorer_formel():
    # Kärnformeln: MAE = (0 + 2)/2 = 1; High-PM MAE (y>=5) = |12-10| = 2
    # => 1 + 1*2 = 3 (lägre = bättre i hjälpfunktionen)
    assert high_pm_score([1.0, 10.0], [1.0, 12.0], 5.0, 1.0) == pytest.approx(3.0)

    # Inga höga värden: fallback på ren MAE
    assert high_pm_score([1.0, 2.0], [2.0, 2.0], 5.0, 1.0) == pytest.approx(0.5)

    # sklearn-scorern (HÖGRE = bättre) med konventionen (estimator, X, y)
    class _Dummy:
        def __init__(self, prediktioner):
            self.prediktioner = prediktioner

        def predict(self, X):
            return self.prediktioner

    scorer = high_pm_scorer(threshold=5.0, lambda_high_pm=1.0)
    assert scorer(_Dummy([1.0, 12.0]), None, [1.0, 10.0]) == pytest.approx(-3.0)
    assert scorer(_Dummy([2.0, 2.0]), None, [1.0, 2.0]) == pytest.approx(-0.5)

    # lambda = 0: sklearn:s inbyggda namn
    assert high_pm_scorer(5.0, 0.0) == "neg_mean_absolute_error"


def test_registrets_standardparametrar_oforandrade():
    """Regressionsspärr: dagens defaults får inte ändras i tysthet."""
    registret = build_model_registry(42)

    assert registret["Ridge Regression"].named_steps["model"].alpha == 1.0
    assert registret["SVR"].named_steps["model"].C == 1.0
    assert registret["SVR"].named_steps["model"].epsilon == 0.1
    assert registret["Random Forest"].n_estimators == 200
    assert registret["Random Forest"].max_depth is None
    assert registret["HistGradientBoosting"].learning_rate == 0.1
    assert registret["HistGradientBoosting"].max_iter == 100
    assert registret["LightGBM"].learning_rate == 0.1
    assert registret["LightGBM"].n_estimators == 100
    assert registret["XGBoost"].learning_rate == 0.3
    assert registret["XGBoost"].n_estimators == 100


def test_registret_med_vunna_params():
    bast = {
        "SVR": {"C": 4.0, "epsilon": 0.05},
        "Random Forest": {"n_estimators": 400, "max_depth": None},
    }
    registret = build_model_registry(42, bast_params=bast)

    assert registret["SVR"].named_steps["model"].C == 4.0
    assert registret["SVR"].named_steps["model"].epsilon == 0.05
    assert registret["Random Forest"].n_estimators == 400
    assert registret["Random Forest"].max_depth is None

    # Familjer utan sökresultat behåller sina standardparametrar
    assert registret["Ridge Regression"].named_steps["model"].alpha == 1.0


def test_sok_basta_params_hittar_param(ren_sok_cache):
    X, y = _sokdata()
    registret = build_model_registry(42)

    bast, info = sok_basta_params(
        "Ridge Regression", registret["Ridge Regression"], X, y,
        grid={"alpha": [0.001, 100.0]},
        horisont=6, n_splits=4, lambda_high_pm=0.0,
    )

    # Stark regularisering (alpha=100) förstör den linjära signalen;
    # sökningen ska välja den svaga regulariseringen.
    assert bast["alpha"] == pytest.approx(0.001)
    assert info["antal_kandidater"] == 2
    assert info["gap"] == 6
    assert info["sokrader"] == len(X)


def test_sok_basta_params_deterministisk(ren_sok_cache):
    X, y = _sokdata()
    registret = build_model_registry(42)

    bast1, info1 = sok_basta_params(
        "Ridge Regression", registret["Ridge Regression"], X, y,
        grid={"alpha": [0.001, 0.1, 1.0, 10.0]},
        horisont=6, n_splits=3, lambda_high_pm=1.0,
    )
    bast2, info2 = sok_basta_params(
        "Ridge Regression", registret["Ridge Regression"], X, y,
        grid={"alpha": [0.001, 0.1, 1.0, 10.0]},
        horisont=6, n_splits=3, lambda_high_pm=1.0,
    )

    assert bast1 == bast2
    assert info1["score"] == pytest.approx(info2["score"])


def test_sok_basta_params_max_rader_anvander_senaste_rader(ren_sok_cache):
    X, y = _sokdata(rader=100)
    registret = build_model_registry(42)

    _, info = sok_basta_params(
        "Ridge Regression", registret["Ridge Regression"], X, y,
        grid={"alpha": [0.001]},
        horisont=6, n_splits=3, lambda_high_pm=0.0,
        max_rader=40,
    )

    assert info["sokrader"] == 40


def test_sok_parametrar_for_registry_delar_mellan_linjer(ren_sok_cache, monkeypatch):
    """Sökresultatet återanvänds mellan modellinjerna inom samma körning."""
    X1, y1 = _sokdata()
    X2, y2 = _sokdata(fro=99)  # annan feature-matris, samma längd

    anrop = {"antal": 0}
    original = tm.sok_basta_params

    def raknare(*args, **kwargs):
        anrop["antal"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(tm, "sok_basta_params", raknare)

    grids = {"Ridge Regression": {"alpha": [0.001, 1.0]}}
    forsta = sok_parametrar_for_registry(X1, y1, grids, horisont=6, n_splits=3)
    andra = sok_parametrar_for_registry(X2, y2, grids, horisont=6, n_splits=3)

    assert anrop["antal"] == 1  # andra linjen återanvänder
    assert forsta == andra
    assert "Ridge Regression" in forsta


def test_sok_parametrar_for_registry_skriver_cache(ren_sok_cache, tmp_path):
    X, y = _sokdata()
    grids = {"Ridge Regression": {"alpha": [0.001, 1.0]}}

    sok_parametrar_for_registry(
        X, y, grids, horisont=6, n_splits=3, models_dir=tmp_path,
    )

    filer = list(tmp_path.glob("hparam_sok_*.json"))
    assert len(filer) == 1

    innehall = json.loads(filer[0].read_text(encoding="utf-8"))
    assert innehall["horisont"] == 6
    assert innehall["resultat"][0]["modell"] == "Ridge Regression"
    assert innehall["resultat"][0]["bast_params"]["alpha"] in (0.001, 1.0)


def test_sok_parametrar_for_registry_failsafe(ren_sok_cache, tmp_path, caplog):
    """
    Fail-safe: ogiltig parameter -> familjen behåller defaults (varning
    loggas), okända modellnamn hoppar över, lyckade familjer söks vidare.
    """
    X, y = _sokdata()
    grids = {
        "Ridge Regression": {"alpha": [0.1], "ogiltig_parameter": [1.0]},
        "Finns Inte": {"C": [1.0]},
        "SVR": {"C": [1.0], "epsilon": [0.1]},
    }

    with caplog.at_level("WARNING", logger="src.train_models"):
        resultat = sok_parametrar_for_registry(
            X, y, grids, horisont=6, n_splits=2, models_dir=tmp_path,
        )

    # Ridge misslyckades (ogiltig parameter), Finns Inte finns inte i
    # registret, men SVR:s sökning lyckades.
    assert "Ridge Regression" not in resultat
    assert "Finns Inte" not in resultat
    assert resultat["SVR"] == {"C": 1.0, "epsilon": 0.1}

    # Varningar loggades för både misslyckandet och det okända namnet
    assert "misslyckades" in caplog.text
    assert "hoppar över" in caplog.text

    # Cachen innehåller bara den lyckade familjen
    innehall = json.loads(
        next(tmp_path.glob("hparam_sok_*.json")).read_text(encoding="utf-8")
    )
    assert len(innehall["resultat"]) == 1
    assert innehall["resultat"][0]["modell"] == "SVR"


# ----------------------------------------------------------
# 4. Nisch-linjen (plan 01): gräns, filtrering, PersistenceModel
# ----------------------------------------------------------

def test_berakna_high_pm_threshold_anvander_pm25_kolumnen():
    """Gränsen beräknas på pm25-kolumnen (värdet vid T — routerns
    nyckel), aldrig på målet y (värdet vid T+horisont)."""
    X_train = pd.DataFrame({"pm25": [1, 2, 3, 4, 5, 6, 7, 8, 9, 20]})
    y_train = pd.Series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109])

    threshold = berakna_high_pm_threshold(X_train)

    # p90 av pm25 = 9 + 0,1*(20-9) = 10,1 (lineär interpolering)
    assert threshold == pytest.approx(10.1)
    # ... inte p90 av y (som vore 108,1)
    assert threshold != pytest.approx(float(y_train.quantile(0.90)))


def test_berakna_high_pm_threshold_opaverkas_av_valideringsdata():
    """Gränsen beräknas ENDAST på träningsdata — extrema
    valideringsvärden får aldrig påverka den."""
    X_train = pd.DataFrame({"pm25": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    X_val = pd.DataFrame({"pm25": [1000, 2000]})

    t1 = berakna_high_pm_threshold(X_train)
    _ = berakna_high_pm_threshold(X_val)  # får inte påverka t1

    assert t1 == pytest.approx(berakna_high_pm_threshold(X_train))
    assert t1 == pytest.approx(9.1)


def test_filtrera_high_pm_behaller_index_och_grans():
    """Filtret behåller rader med pm25 >= gränsen (inklusivt, >=, som
    routern) och X/y filtreras med SAMMA mask."""
    X_train = pd.DataFrame({"pm25": [5, 15, 25, 35], "ovrig": [1, 2, 3, 4]})
    y_train = pd.Series([6, 16, 26, 36])
    X_val = pd.DataFrame({"pm25": [1, 20, 30], "ovrig": [7, 8, 9]})
    y_val = pd.Series([2, 21, 31])

    X_tr, y_tr, X_v, y_v = filtrera_high_pm(
        X_train, y_train, X_val, y_val, threshold=15,
    )

    assert list(X_tr["pm25"]) == [15, 25, 35]
    assert list(y_tr) == [16, 26, 36]          # samma rader som X
    assert list(X_tr.index) == list(y_tr.index)  # indexen hänger ihop
    assert list(X_v["pm25"]) == [20, 30]
    assert list(y_v) == [21, 31]
    # Gränsraden selv ingår (>=)
    X_tr2, *_ = filtrera_high_pm(X_train, y_train, X_val, y_val, threshold=25)
    assert list(X_tr2["pm25"]) == [25, 35]


def test_train_and_evaluate_models_med_explicit_grans():
    """Explicit gräns (plan 01) styrs High-måtten — p90 av y_train
    ignoreras."""
    X_train, y_train, X_val, y_val = sample_data = (
        pd.DataFrame({"pm25": [10, 15, 20, 25, 30, 40, 50, 60, 70, 80]}),
        pd.Series([12, 17, 22, 27, 32, 42, 52, 62, 72, 82]),
        pd.DataFrame({"pm25": [15, 30, 50, 75]}),
        pd.Series([17, 32, 52, 77]),
    )

    results_df, _ = train_and_evaluate_models(
        {"Linear Regression": LinearRegression()},
        X_train, y_train, X_val, y_val,
        high_pm25_threshold=50,
    )

    # y_val >= 50: 52 och 77 -> räkning 2 (default p90 av y_train = 73
    # hade gett räkning 1)
    assert results_df.iloc[0]["Validation High PM2.5 Count"] == 2

    # Modellen lär sig y = pm25 + 2
    assert results_df.iloc[0]["Validation MAE"] == pytest.approx(0)


def test_persistence_baseline_med_explicit_grans(sample_data):
    """Baslinjen utvärderas med turneringens explicita gräns."""
    X_train, y_train, X_val, y_val = sample_data

    _, val_metrics = persistence_baseline(
        X_train, y_train, X_val, y_val, high_pm25_threshold=17,
    )

    # y_val >= 17: alla fyra rader; baslinjen missar med 2
    assert val_metrics["High PM2.5 Count"] == 4
    assert val_metrics["High PM2.5 MAE"] == pytest.approx(2.0)


def test_persistence_model_predikterar_nuvaerdet():
    """PersistenceModel 'predikterar' att pm25 består — och kan fit:as
    som vilken estimator som helst (krav för turneringen och joblib)."""
    X = pd.DataFrame({"pm25": [3.0, 14.4, 25.0], "ovrig": [1, 2, 3]})

    modell = PersistenceModel().fit(X)

    assert modell is not None
    assert list(modell.predict(X)) == [3.0, 14.4, 25.0]
    # Enradig DataFrame (live-fallet) fungerar också
    assert modell.predict(X.head(1))[0] == pytest.approx(3.0)


# ----------------------------------------------------------
# 5. Kompositscore-kolumnen (plan 02)
# ----------------------------------------------------------

VIKTER = {
    "w_mae": 0.5,
    "w_high_mae": 0.35,
    "w_high_bias": 0.15,
    "min_antal_high": 10,
}


class _HighFelModell:
    """Predikterar pm25 + 5 på rader med pm25 >= 20, annars pm25 —
    olika fel över och under tröskeln, så kompositens tre termer
    skiljer sig åt och kan handräknas."""

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        pm25 = X["pm25"].to_numpy()
        return pm25 + np.where(pm25 >= 20, 5.0, 0.0)


def test_train_and_evaluate_models_beraknar_komposit_handraknat():
    """'Validation Composite' = 0,5·MAE + 0,35·High MAE + 0,15·|High Bias|,
    handräknad på en fixture där alla High-villkor är uppfyllda."""
    # val: pm25 1..30, target = pm25 + 2. Utvärderingsgräns 20 => de
    # 13 raderna med pm25 >= 18 (y >= 20) utgör High-PM-underlaget.
    X_val = pd.DataFrame({"pm25": list(range(1, 31))})
    y_val = pd.Series([v + 2 for v in X_val["pm25"]])
    X_train = pd.DataFrame({"pm25": list(range(1, 41))})
    y_train = pd.Series([v + 2 for v in X_train["pm25"]])

    results_df, _ = train_and_evaluate_models(
        {"HighFel": _HighFelModell()},
        X_train, y_train, X_val, y_val,
        high_pm25_threshold=20,
        vikter=VIKTER,
    )

    rad = results_df.iloc[0]

    # Handräkning:
    # MAE = (19 rader med fel −2 [pm25 1..19] + 11 rader med fel +3
    #        [pm25 20..30]) / 30 = (38 + 33)/30 = 71/30
    # High-slice (y >= 20, dvs pm25 >= 18): fel = −2, −2, +3×11
    #        High MAE = (4 + 33)/13 = 37/13
    #        High Bias = (−4 + 33)/13 = 29/13
    # Composite = 0,5·71/30 + 0,35·37/13 + 0,15·29/13
    exp_mae = 71 / 30
    exp_high_mae = 37 / 13
    exp_high_bias = 29 / 13
    exp_composite = (
        0.5 * exp_mae + 0.35 * exp_high_mae + 0.15 * abs(exp_high_bias)
    )

    assert rad["Validation MAE"] == pytest.approx(exp_mae)
    assert rad["Validation High PM2.5 MAE"] == pytest.approx(exp_high_mae)
    assert rad["Validation High PM2.5 Bias"] == pytest.approx(exp_high_bias)
    assert rad["Validation Composite"] == pytest.approx(exp_composite)


def test_train_and_evaluate_models_komposit_faller_tillbaka_pa_mae(caplog):
    """För få High-rader (under min_antal_high) -> 'Validation Composite'
    degraderar till ren MAE (med varning, se composite_score)."""
    X_train = pd.DataFrame({"pm25": list(range(1, 41))})
    y_train = pd.Series([v + 2 for v in X_train["pm25"]])
    X_val = pd.DataFrame({"pm25": list(range(1, 31))})
    y_val = pd.Series([v + 2 for v in X_val["pm25"]])

    # Gräns 31 => inga val-rader över gränsen => NaN High-mått i val.
    with caplog.at_level("WARNING", logger="src.metrics"):
        results_df, _ = train_and_evaluate_models(
            {"HighFel": _HighFelModell()},
            X_train, y_train, X_val, y_val,
            high_pm25_threshold=31,
            vikter=VIKTER,
        )

    rad = results_df.iloc[0]

    assert rad["Validation Composite"] == pytest.approx(rad["Validation MAE"])
    assert "High-PM-underlag" in caplog.text


def test_train_and_evaluate_models_komposit_utan_vikter_anvander_default():
    """Utan vikter används standardvikterna (metrics.DEFAULT_COMPOSITE_VIKTER)
    — kolumnen beräknas alltid, så results_df:n blir förutsägbart formad."""
    X_train = pd.DataFrame({"pm25": list(range(1, 41))})
    y_train = pd.Series([v + 2 for v in X_train["pm25"]])
    X_val = pd.DataFrame({"pm25": list(range(101, 131))})
    y_val = pd.Series([v + 2 for v in X_val["pm25"]])

    with_vikter, _ = train_and_evaluate_models(
        {"HighFel": _HighFelModell()},
        X_train, y_train, X_val, y_val,
        high_pm25_threshold=20,
        vikter={
            "w_mae": 0.5, "w_high_mae": 0.35, "w_high_bias": 0.15,
            "min_antal_high": 10,
        },
    )
    utan_vikter, _ = train_and_evaluate_models(
        {"HighFel": _HighFelModell()},
        X_train, y_train, X_val, y_val,
        high_pm25_threshold=20,
    )

    assert (
        utan_vikter.iloc[0]["Validation Composite"]
        == pytest.approx(with_vikter.iloc[0]["Validation Composite"])
    )

