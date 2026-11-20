import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from src.metrics import evaluate_model, composite_score
from src.logger import setup_logger

from sklearn.base import clone
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

log = setup_logger(__name__)


def build_model_registry(random_state=42, bast_params=None):
    """
    Returnerar en dict med namn -> (estimator) för alla modeller
    som ska tävla i modellturneringen.

    Parametrar
    ----------
    random_state : int
        Slumpvärde för reproducerbarhet hos modeller som stödjer det.
    bast_params : dict, valfri
        namn -> dict med vunna sökparametrar (från
        sok_parametrar_for_registry). Modellfamiljer utan sökresultat
        initieras med samma standardparametrar som tidigare. Saknade
        enskilda parametrar faller tillbaka på respektive standardvärde,
        så ett ofullständigt sökresultat kan aldrig krascha träningen.

    Returnerar
    -------
    dict[str, estimator]
    """
    # Vunna parametrar per familj; {} för familjer utan sökresultat
    def familj(namn: str) -> dict:
        return (bast_params or {}).get(namn, {})

    svr = familj("SVR")
    ridge = familj("Ridge Regression")
    rf = familj("Random Forest")
    hgb = familj("HistGradientBoosting")
    lgbm = familj("LightGBM")
    xgb = familj("XGBoost")

    return {
        "Linear Regression": LinearRegression(),

        "Ridge Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=ridge.get("alpha", 1.0))),
        ]),

        "Random Forest": RandomForestRegressor(
            n_estimators=rf.get("n_estimators", 200),
            max_depth=rf.get("max_depth"),
            random_state=random_state,
            n_jobs=-1,
        ),

        "HistGradientBoosting": HistGradientBoostingRegressor(
            learning_rate=hgb.get("learning_rate", 0.1),
            max_iter=hgb.get("max_iter", 100),
            random_state=random_state,
        ),

        "SVR": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVR(
                kernel="rbf",
                C=svr.get("C", 1.0),
                epsilon=svr.get("epsilon", 0.1),
            )),
        ]),

        "LightGBM": LGBMRegressor(
            learning_rate=lgbm.get("learning_rate", 0.1),
            n_estimators=lgbm.get("n_estimators", 100),
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        ),

        "XGBoost": XGBRegressor(
            learning_rate=xgb.get("learning_rate", 0.3),
            n_estimators=xgb.get("n_estimators", 100),
            random_state=random_state,
            n_jobs=-1,
            verbosity=0,
        ),
    }


class PersistenceModel:
    """
    Persistence-baseline som "riktig" modell (plan 01): predikterar att
    PM2.5 om sex timmar är samma som det aktuella värdet, dvs. kolumnen
    'pm25' i indatan.

    Wrappern behövs för nisch-linjen: vinner baslinjen nisch-turneringen
    måste den kunna sparas som champion (.joblib) och servas av live-
    routern exakt som en vanlig modell — annars skulle den bästa
    högdatalösningen aldrig nå drift (beslut 2026-11-15).

    OBS: klassen måste vara definierad på modulnivå i ett importerbart
    module, annars hittar inte joblib.load den vid inläsning.
    """

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return X["pm25"].to_numpy()


def berakna_high_pm_threshold(X_train):
    """
    Gränsen för 'ovanligt hög' PM2.5 (plan 01): 90:e percentilen av
    träningsdatans pm25 — dvs. värdet vid T, samma kolumn som live-
    routern nycklar på (INTE målet y vid T+horisont). Beräknas alltid
    på det FULLSTÄNDIGA träningsunderlaget, innan eventuell filtrering.

    Returnerar
    -------
    float
    """
    return float(X_train["pm25"].quantile(0.90))


def filtrera_high_pm(X_train, y_train, X_val, y_val, threshold):
    """
    Filtrerar tränings- och valideringsrader till pm25 >= threshold —
    samma kolumn ('pm25', värdet vid T) och samma gräns som routern,
    nisch-träningen och utvärderingen använder (ingen mismatch).

    X och y filtreras med samma mask, så raderna hänger ihop.

    Returnerar
    -------
    tuple
        (X_train_hoga, y_train_hoga, X_val_hoga, y_val_hoga).
    """
    mask_train = X_train["pm25"] >= threshold
    mask_val = X_val["pm25"] >= threshold

    return (
        X_train.loc[mask_train],
        y_train.loc[mask_train],
        X_val.loc[mask_val],
        y_val.loc[mask_val],
    )


def train_and_evaluate_models(models, X_train, y_train, X_val, y_val, high_pm25_threshold=None, vikter=None):
    """
    Tränar och utvärderar samtliga modeller.

    Utvärderingen omfattar MAE, RMSE, R², Bias samt
    särskilda mått för höga PM2.5-koncentrationer.

    Gränsen för höga PM2.5-värden definieras som
    90:e percentilen av träningsdata. high_pm25_threshold kan anges
    explicit (plan 01: gränsen beräknas på pm25-kolumnen i FULLA
    träningsdata och delas av filter, turnering och router).

    Plan 02: per modell beräknas även kolumnen "Validation Composite" —
    den viktade kompositscoren (se composite_score i src/metrics.py)
    som kandidatvalet och championduellen avgörs efter.

    Parametrar
    ----------
    models : dict[str, estimator]
        Modeller som ska tränas och jämföras.
    X_train, X_val : pd.DataFrame
        Förklarande variabler.
    y_train, y_val : pd.Series
        Faktiska målvärden.
    high_pm25_threshold : float, valfri
        Explicit gräns för höga PM2.5-värden. None => p90 av y_train
        (tidigare beteende).
    vikter : dict, valfri
        Vikter till kompositscoren ([model.composite] i config.toml).
        None => standardvikterna i metrics.DEFAULT_COMPOSITE_VIKTER.

    Returnerar
    -------
    tuple[pd.DataFrame, dict]
        Resultattabell och tränade modeller.
    """

    # 1. Förbered resultat och modeller
    results = []
    trained_models = {}

    # 2. Beräkna gemensam gräns från träningsdata
    if high_pm25_threshold is None:
        high_pm25_threshold = y_train.quantile(0.90)

    print(
        f"High PM2.5 threshold: "
        f"{high_pm25_threshold:.2f} µg/m³"
    )

    # 3. Träna och utvärdera varje modell
    for name, model in models.items():
        print(f"Tränar {name}...")

        model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)

        # 4. Beräkna samtliga utvärderingsmått
        train_metrics = evaluate_model(
            y_train, y_pred_train, threshold=high_pm25_threshold
        )

        val_metrics = evaluate_model(
            y_val, y_pred_val, threshold=high_pm25_threshold
        )

        # 5. Sammanställ resultat
        results.append({
            "Modell": name,
            **{
                f"Train {k.replace('R2', 'R²')}": v
                for k, v in train_metrics.items()
            },
            **{
                f"Validation {k.replace('R2', 'R²')}": v
                for k, v in val_metrics.items()
            },
            # Plan 02: kompositscore per modell (lägre = bättre)
            "Validation Composite": composite_score(val_metrics, vikter),
        })

        trained_models[name] = model

    # 6. Skapa och sortera resultattabell
    results_df = (
        pd.DataFrame(results)
        .sort_values(by="Validation MAE", ascending=True)
        .reset_index(drop=True)
    )

    return results_df, trained_models



def load_train_val_data(
    train_path,
    val_path,
    features,
    target,
):
    """
    Läser in processad tränings- och valideringsdata och delar upp i X och y.

    Parametrar
    ----------
    train_path : Path
        Sökväg till processad träningsdata (cfg.train_processed).
    val_path : Path
        Sökväg till processad valideringsdata (cfg.val_processed).
    features : list[str]
        Kolumner som ska användas som features (cfg.features).
    target : str
        Namn på målvariabeln, t.ex. "target".

    Returnerar
    -------
    (X_train, y_train, X_val, y_val) : tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]
    """
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    X_train = train_df[features]
    y_train = train_df[target]

    X_val = val_df[features]
    y_val = val_df[target]

    return X_train, y_train, X_val, y_val


def persistence_baseline(X_train, y_train, X_val, y_val, high_pm25_threshold=None):
    """
    Persistence-baseline: antar att framtida PM2.5
    är lika med det aktuella värdet.

    Utvärderas med samma mått och gränsvärde
    som övriga modeller.

    high_pm25_threshold kan anges explicit (plan 01) — samma gräns som
    turneringen och routern, istället för p90 av y_train.

    Returnerar
    -------
    tuple[dict, dict]
        Utvärderingsmått för träning och validering.
    """

    # 1. Beräkna samma gränsvärde
    if high_pm25_threshold is None:
        high_pm25_threshold = y_train.quantile(0.90)

    # 2. Persistence-prediktioner
    y_pred_train = X_train["pm25"]
    y_pred_val = X_val["pm25"]

    # 3. Utvärdera baseline
    train_metrics = evaluate_model(
        y_train, y_pred_train, threshold=high_pm25_threshold
    )

    val_metrics = evaluate_model(
        y_val, y_pred_val, threshold=high_pm25_threshold
    )

    return train_metrics, val_metrics


#----------------------------------------------------------
# Hyperparametersökning (plan 05, beslut 2026-11-15)
#----------------------------------------------------------
# Sökningen körs ENDAST i run_model_line vid omträning (championens
# 7-dagarsklocka) — aldrig i live-prediktioner. Resultatet delas
# medvetet mellan den vanliga och den förskjutna modellinjen inom
# samma pipeline-körning (approximation för att halvera kostnaden).

# Minnescache: nyckel -> vunna parametrar. Nollställs per process,
# dvs. en full pipeline-körning söker varje familj högst en gång.
_SOK_MEMO: dict = {}

# Cache-filen skrivs en gång per process (andra modellinjen återanvänder
# samma innehåll, se spara_hparam_cache).
_HPARAM_CACHE_SKRIVEN = False


def high_pm_score(y_true, y_pred, threshold: float, lambda_high_pm: float) -> float:
    """
    Sök-scorens kärna (LÄGRE = bättre, sklearn-konvention inverteras
    av anroparen):

        MAE + lambda_high_pm · MAE bland rader med y_true >= threshold

    Saknas värden över gränsen returneras ren MAE.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    bas_mae = mean_absolute_error(y_true, y_pred)
    hoga = y_true >= threshold

    if not hoga.any():
        return bas_mae

    return bas_mae + lambda_high_pm * mean_absolute_error(
        y_true[hoga], y_pred[hoga]
    )


def high_pm_scorer(threshold: float, lambda_high_pm: float):
    """
    Returnerar en sklearn-scorer (HÖGRE = bättre) för sökningen, enligt
    sklearn-konventionen scorer(estimator, X, y) (gäller sedan 1.6):

        score = -(MAE + lambda_high_pm · High-PM MAE)

    lambda_high_pm <= 0 ger sklearn:s inbyggda neg_mean_absolute_error
    istället. Se high_pm_score för beräkningen.
    """
    if lambda_high_pm <= 0:
        return "neg_mean_absolute_error"

    def scorer(estimator, X, y):
        return -high_pm_score(y, estimator.predict(X), threshold, lambda_high_pm)

    scorer.__name__ = "high_pm_viktad_mae"
    return scorer


def _oversatt_grid(grid: dict, estimator) -> dict:
    """
    Översätter en config-grid ("bare" parameternamn) till GridSearchCV-
    format: Pipeline-estimatorer (Ridge, SVR) kräver "model__"-prefix.
    Strängvärdet "none" översätts till Python-None (t.ex. max_depth).
    """
    ar_pipeline = hasattr(estimator, "named_steps")

    ut = {}
    for namn, vardeslista in grid.items():
        nyckel = f"model__{namn}" if ar_pipeline else namn
        ut[nyckel] = [None if varde == "none" else varde for varde in vardeslista]

    return ut


def sok_basta_params(
    modell_namn,
    estimator,
    X_train,
    y_train,
    grid,
    horisont,
    n_splits=4,
    lambda_high_pm=0.0,
    max_rader=0,
):
    """
    Söker bästa hyperparametrar för EN modellfamilj med kronologisk
    korsvalidering: TimeSeriesSplit med gap=horisont, så att tränings-
    faldens mål (T+horisont) aldrig hamnar i sitt valdfald — inget
    måldataläckage. Sökningen sker ENDAST på träningsperioden; valide-
    rings- och testdata berörs inte av sökningen (de används senare av
    turneringen, som tidigare).

    Score: se high_pm_scorer. Threshold beräknas som p90 av y i
    sökunderlaget (träningsdata).

    max_rader > 0: söker på de SENASTE max_rader raderna — approximativ
    genväg för dyra familjer (SVR). Vinnande parametrar tränas sedan av
    turneringen på det fullständiga träningsunderlaget.

    Returnerar (bast_params, info), där bast_params använder registrets
    "bare" parameternamn (utan pipeline-prefix). Raiser vid fel —
    fail-safe hanteras av sok_parametrar_for_registry.
    """
    t0 = time.perf_counter()

    if max_rader and len(X_train) > max_rader:
        X_sok = X_train.iloc[-max_rader:]
        y_sok = y_train.iloc[-max_rader:]
        log.info(
            "Hparam-sökning för %s: använder de %s senaste av %s träningsraderna.",
            modell_namn, len(X_sok), len(X_train),
        )
    else:
        X_sok, y_sok = X_train, y_train

    threshold = float(pd.Series(y_sok).quantile(0.90))
    scorer = high_pm_scorer(threshold, lambda_high_pm)
    cv = TimeSeriesSplit(n_splits=n_splits, gap=horisont)

    # Klonad estimator med n_jobs=1: parallelliseringen sköts på
    # GridSearchCV-nivån (annars byggs kapslade trådpooler).
    sok_estimator = clone(estimator)
    if hasattr(sok_estimator, "named_steps"):
        modell_steg = sok_estimator.named_steps.get("model")
        if hasattr(modell_steg, "n_jobs"):
            modell_steg.n_jobs = 1
    elif hasattr(sok_estimator, "n_jobs"):
        sok_estimator.n_jobs = 1

    gs = GridSearchCV(
        sok_estimator,
        _oversatt_grid(grid, sok_estimator),
        scoring=scorer,
        cv=cv,
        n_jobs=-1,
        refit=False,
    )
    gs.fit(X_sok, y_sok)

    bast_index = gs.best_index_
    score = float(gs.cv_results_["mean_test_score"][bast_index])
    if np.isnan(score):
        raise ValueError(f"Alla kandidater misslyckades för {modell_namn}.")

    bast = gs.cv_results_["params"][bast_index]
    bast_bare = {k.removeprefix("model__"): v for k, v in bast.items()}

    info = {
        "modell": modell_namn,
        "score": score,
        "scorer": (
            "negativ MAE" if lambda_high_pm <= 0
            else f"-(MAE + {lambda_high_pm}·High-PM MAE)"
        ),
        "high_pm_threshold": threshold,
        "antal_kandidater": len(gs.cv_results_["params"]),
        "n_splits": n_splits,
        "gap": horisont,
        "sokrader": int(len(X_sok)),
        "soktid_s": round(time.perf_counter() - t0, 1),
        "bast_params": bast_bare,
    }

    return bast_bare, info


def sok_parametrar_for_registry(
    X_train,
    y_train,
    grids,
    horisont,
    n_splits=4,
    lambda_high_pm=0.0,
    sok_rader_per_modell=None,
    models_dir=None,
):
    """
    Kör sok_basta_params per modellfamilj, med fail-safe: misslyckas
    en familjs sökning behåller den sina standardparametrar (varning
    loggas) och övriga familjer söks vidare.

    Minnescachen gör att den förskjutna modellinjen (som anropar med
    samma träningsunderlag i y, men förskjutna väderfeatures) åter-
    använder samma sökresultat som den vanliga linjen — sökresultatet
    delas medvetet mellan linjerna inom samma körning, se beslut
    2026-11-15 (kostnad). En ny process söker alltid på nytt.

    grids : dict
        namn -> param-grid från [model.hparam_search.grid] i
        config.toml.
    sok_rader_per_modell : dict, valfri
        namn -> max antal rader för sökunderlaget (t.ex. SVR).
    models_dir : Path, valfri
        Skrivs den, cacheas resultatet till
        models/hparam_sok_<timestamp>.json (en gång per process).

    Returnerar dict namn -> bast_params (bara lyckade sökningar).
    """
    global _HPARAM_CACHE_SKRIVEN
    sok_rader_per_modell = sok_rader_per_modell or {}
    bast_params = {}
    info_rader = []
    standard = build_model_registry()

    for namn, grid in grids.items():
        if namn not in standard:
            log.warning(
                "Hparam-sökningen: modellnamnet '%s' finns inte i "
                "modellregistret – hoppar över.", namn,
            )
            continue

        max_rader = sok_rader_per_modell.get(namn, 0)
        nyckel = (
            namn,
            tuple(sorted((k, tuple(v)) for k, v in grid.items())),
            len(y_train),
            n_splits,
            lambda_high_pm,
            max_rader,
            horisont,
        )

        if nyckel in _SOK_MEMO:
            log.info(
                "Återanvänder sökresultatet för %s (delas mellan "
                "modellinjerna inom körningen).", namn,
            )
            bast_params[namn] = _SOK_MEMO[nyckel]
            continue

        try:
            params, info = sok_basta_params(
                modell_namn=namn,
                estimator=standard[namn],
                X_train=X_train,
                y_train=y_train,
                grid=grid,
                horisont=horisont,
                n_splits=n_splits,
                lambda_high_pm=lambda_high_pm,
                max_rader=max_rader,
            )
        except Exception:
            log.warning(
                "Hparam-sökningen misslyckades för %s – modellen behåller "
                "sina standardparametrar.", namn, exc_info=True,
            )
            continue

        _SOK_MEMO[nyckel] = params
        bast_params[namn] = params
        info_rader.append(info)

        log.info(
            "Bästa params för %s: %s (sök-score %.4f på %s rader, %s s).",
            namn, params, info["score"], info["sokrader"], info["soktid_s"],
        )

    if models_dir is not None and info_rader:
        spara_hparam_cache(models_dir, info_rader, horisont, n_splits, lambda_high_pm)

    return bast_params


def spara_hparam_cache(models_dir, info_rader, horisont, n_splits, lambda_high_pm):
    """
    Skriver sökresultatet till models/hparam_sok_<timestamp>.json, så
    att varje parametrbyte går att spåra i efterhand. Skrivs en gång
    per process (andra modellinjen återanvänder samma cache-innehåll).
    """
    global _HPARAM_CACHE_SKRIVEN
    if _HPARAM_CACHE_SKRIVEN:
        return

    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    fil = models_dir / f"hparam_sok_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.json"
    innehall = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "horisont": horisont,
        "n_splits": n_splits,
        "lambda_high_pm": lambda_high_pm,
        "notering": (
            "Sökresultatet delas mellan den vanliga och den förskjutna "
            "modellinjen inom samma pipeline-körning (beslut 2026-11-15)."
        ),
        "resultat": info_rader,
    }

    with open(fil, "w", encoding="utf-8") as f:
        json.dump(innehall, f, indent=2, ensure_ascii=False, default=float)

    log.info("Sökresultaten är cachade i %s.", fil)
    _HPARAM_CACHE_SKRIVEN = True

