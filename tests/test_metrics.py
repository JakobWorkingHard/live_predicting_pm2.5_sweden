import pytest
import pandas as pd

from types import SimpleNamespace

from src.metrics import (
    evaluate_model,
    build_comparison_table,
    select_best_model,
    metric_direction,
    composite_score,
    jamfor_championer,
    jamfor_routat_system,
)


def test_evaluate_model_without_threshold():
    """Kontrollerar grundläggande mått inklusive Bias."""

    y_true = [10, 20, 30, 40, 50]
    y_pred = [12, 18, 25, 35, 40]

    result = evaluate_model(y_true, y_pred)

    assert result["MAE"] == pytest.approx(4.8)
    assert result["RMSE"] == pytest.approx(5.62, abs=0.01)
    assert result["R2"] == pytest.approx(0.842)
    assert result["Bias"] == pytest.approx(-4.0)


def test_evaluate_model_with_threshold():
    """Kontrollerar utvärdering vid höga PM2.5-värden."""

    y_true = [10, 20, 30, 40, 50]
    y_pred = [12, 18, 25, 35, 40]

    result = evaluate_model(y_true, y_pred, threshold=40)

    assert result["High PM2.5 MAE"] == pytest.approx(7.5)
    assert result["High PM2.5 Bias"] == pytest.approx(-7.5)
    assert result["High PM2.5 Count"] == 2


def test_evaluate_model_without_high_values():
    """Kontrollerar att inga höga observationer hanteras."""

    y_true = [10, 20, 30]
    y_pred = [12, 18, 25]

    result = evaluate_model(y_true, y_pred, threshold=100)

    assert result["High PM2.5 Count"] == 0
    assert result["High PM2.5 MAE"] != result["High PM2.5 MAE"]
    assert result["High PM2.5 Bias"] != result["High PM2.5 Bias"]


def test_build_comparison_table():
    """Kontrollerar att modeller och baseline jämförs korrekt."""

    results_df = pd.DataFrame([
        {
            "Modell": "Random Forest",
            "Validation MAE": 3.0,
            "Validation RMSE": 4.5,
            "Validation R²": 0.80,
            "Validation Bias": -1.0,
            "Validation High PM2.5 MAE": 6.0,
        },
        {
            "Modell": "Linear Regression",
            "Validation MAE": 4.0,
            "Validation RMSE": 5.0,
            "Validation R²": 0.70,
            "Validation Bias": -2.0,
            "Validation High PM2.5 MAE": 8.0,
        },
    ])

    baseline_metrics = {
        "MAE": 5.0,
        "RMSE": 6.0,
        "R2": 0.60,
        "Bias": -3.0,
        "High PM2.5 MAE": 10.0,
    }

    comparison = build_comparison_table(
        results_df,
        baseline_metrics,
        sort_by="Validation MAE",
    )

    assert len(comparison) == 3
    assert comparison.iloc[0]["Modell"] == "Random Forest"
    assert "Validation Bias" in comparison.columns
    assert "Validation High PM2.5 MAE" in comparison.columns

    baseline = comparison[
        comparison["Modell"] == "Persistence Baseline"
    ].iloc[0]

    assert baseline["Validation MAE"] == 5.0
    assert baseline["Validation Bias"] == -3.0
    assert baseline["Validation High PM2.5 MAE"] == 10.0
    assert baseline["Validation R²"] == pytest.approx(0.60)


def test_build_comparison_table_sort_r2():
    """Kontrollerar att högsta R² hamnar först."""

    results_df = pd.DataFrame([
        {
            "Modell": "Model A",
            "Validation MAE": 3.0,
            "Validation R²": 0.70,
        },
        {
            "Modell": "Model B",
            "Validation MAE": 2.0,
            "Validation R²": 0.90,
        },
    ])

    baseline_metrics = {
        "MAE": 4.0,
        "R2": 0.50,
    }

    comparison = build_comparison_table(
        results_df,
        baseline_metrics,
        sort_by="Validation R²",
    )

    assert comparison.iloc[0]["Modell"] == "Model B"


def test_select_best_model_mae():
    """Lägsta MAE ska vinna."""

    results_df = pd.DataFrame([
        {"Modell": "Model A", "Validation MAE": 3.0},
        {"Modell": "Model B", "Validation MAE": 2.0},
    ])

    name, row = select_best_model(results_df, "Validation MAE")

    assert name == "Model B"
    assert row["Validation MAE"] == 2.0


def test_select_best_model_r2():
    """Högsta R² ska vinna."""

    results_df = pd.DataFrame([
        {"Modell": "Model A", "Validation R²": 0.90},
        {"Modell": "Model B", "Validation R²": 0.70},
    ])

    name, row = select_best_model(results_df, "Validation R²")

    assert name == "Model A"


def test_select_best_model_bias():
    """Bias närmast noll ska vinna."""

    results_df = pd.DataFrame([
        {"Modell": "Model A", "Validation Bias": -5.0},
        {"Modell": "Model B", "Validation Bias": 1.0},
        {"Modell": "Model C", "Validation Bias": 3.0},
    ])

    name, row = select_best_model(results_df, "Validation Bias")

    assert name == "Model B"


def test_build_comparison_table_sort_bias():
    """Bias närmast noll ska hamna först."""

    results_df = pd.DataFrame([
        {"Modell": "Model A", "Validation Bias": -5.0},
        {"Modell": "Model B", "Validation Bias": 1.0},
    ])

    baseline_metrics = {"Bias": -2.0}

    comparison = build_comparison_table(
        results_df,
        baseline_metrics,
        sort_by="Validation Bias",
    )

    assert comparison["Modell"].tolist() == [
        "Model B",
        "Persistence Baseline",
        "Model A",
    ]


def test_jamfor_championer_med_high_pm25(tmp_path, monkeypatch):

    class DummyModel:
        def predict(self, X):
            return X["pm25"].to_numpy() + 2

    # Två testfiler med samma tidpunkter
    df = pd.DataFrame({
        "tid": ["2026-06-01 00:00", "2026-06-01 01:00",
                "2026-06-01 02:00"],
        "pm25": [10, 30, 50],
        "target": [12, 32, 52],
    })

    vanlig_path = tmp_path / "vanlig.csv"
    forskjuten_path = tmp_path / "forskjuten.csv"

    df.to_csv(vanlig_path, index=False)
    df.to_csv(forskjuten_path, index=False)

    cfg = SimpleNamespace(
        features=["pm25"],
        target_column="target",
        jamfordelse_log=tmp_path / "jamfordelse.csv",
    )

    # Fånga resultatet utan att skriva någon riktig logg
    sparade_resultat = []

    monkeypatch.setattr(
        "src.metrics.append_jamfordelse_log",
        lambda path, df: sparade_resultat.append(df.copy()),
    )

    jamfor_championer(
        cfg=cfg,
        champion_modell=DummyModel(),
        champion_namn="Champion",
        forskjuten_modell=DummyModel(),
        forskjuten_namn="Förskjuten",
        test_path_vanlig=vanlig_path,
        test_path_forskjuten=forskjuten_path,
        high_pm25_threshold=40,
    )

    assert len(sparade_resultat) == 1

    resultat = sparade_resultat[0]

    # Tre modeller ska jämföras
    assert len(resultat) == 3

    # Kontrollera att de nya måtten finns
    assert "Bias" in resultat.columns
    assert "High PM2.5 MAE" in resultat.columns
    assert "High PM2.5 Bias" in resultat.columns
    assert "High PM2.5 Count" in resultat.columns

    # Bara ett faktiskt PM2.5-värde är >= 40
    assert (resultat["High PM2.5 Count"] == 1).all()

    # Persistence baseline underskattar med 2
    baseline = resultat.loc[
        resultat["Modell"] == "Persistence Baseline"
    ].iloc[0]

    assert baseline["Bias"] == pytest.approx(-2.0)
    assert baseline["High PM2.5 MAE"] == pytest.approx(2.0)
    assert baseline["High PM2.5 Bias"] == pytest.approx(-2.0)


def test_jamfor_championer_utan_threshold(tmp_path, monkeypatch):

    class DummyModel:
        def predict(self, X):
            return X["pm25"].to_numpy() + 2

    df = pd.DataFrame({
        "tid": ["2026-06-01 00:00", "2026-06-01 01:00"],
        "pm25": [10, 30],
        "target": [12, 32],
    })

    vanlig_path = tmp_path / "vanlig.csv"
    forskjuten_path = tmp_path / "forskjuten.csv"

    df.to_csv(vanlig_path, index=False)
    df.to_csv(forskjuten_path, index=False)

    cfg = SimpleNamespace(
        features=["pm25"],
        target_column="target",
        jamfordelse_log=tmp_path / "jamfordelse.csv",
    )

    sparade_resultat = []

    monkeypatch.setattr(
        "src.metrics.append_jamfordelse_log",
        lambda path, df: sparade_resultat.append(df.copy()),
    )

    # Observera att high_pm25_threshold INTE skickas med
    jamfor_championer(
        cfg=cfg,
        champion_modell=DummyModel(),
        champion_namn="Champion",
        forskjuten_modell=DummyModel(),
        forskjuten_namn="Förskjuten",
        test_path_vanlig=vanlig_path,
        test_path_forskjuten=forskjuten_path,
    )

    assert len(sparade_resultat) == 1

    resultat = sparade_resultat[0]

    assert len(resultat) == 3

    # Grundläggande mått ska finnas
    assert "MAE" in resultat.columns
    assert "RMSE" in resultat.columns
    assert "R2" in resultat.columns
    assert "Bias" in resultat.columns

    # Specialiserade mått kräver ett gränsvärde
    assert "High PM2.5 MAE" not in resultat.columns
    assert "High PM2.5 Bias" not in resultat.columns
    assert "High PM2.5 Count" not in resultat.columns


def test_jamfor_routat_system(tmp_path, monkeypatch):
    """Plan 01: routat system (champion under gränsen, nisch-champion
    vid/över) mot championen på valideringsdata — med 'MAE under
    gräns' som bevis på att routern lämnar raderna under gränsen orörda."""

    class ChampionDummy:
        def predict(self, X):
            return X["pm25"].to_numpy() + 0.5

    class NischDummy:
        def predict(self, X):
            return X["pm25"].to_numpy() + 2.0

    # pm25 [1, 2, 3] under gränsen 5; [10, 11] på/över. target = pm25
    # (persistence perfekt).
    val_df = pd.DataFrame({
        "pm25": [1, 2, 3, 10, 11],
        "target": [1, 2, 3, 10, 11],
    })
    val_path = tmp_path / "val.csv"
    val_df.to_csv(val_path, index=False)

    cfg = SimpleNamespace(
        features=["pm25"],
        target_column="target",
        jamfordelse_log=tmp_path / "jamforelse.csv",
    )

    sparade_resultat = []
    monkeypatch.setattr(
        "src.metrics.append_jamfordelse_log",
        lambda path, df: sparade_resultat.append(df.copy()),
    )

    jamfor_routat_system(
        cfg=cfg,
        champion_modell=ChampionDummy(),
        champion_namn="SVR vanlig (T+6)",
        niche_modell=NischDummy(),
        niche_namn="SVR nisch (T+6)",
        niche_threshold=5.0,
        val_path=val_path,
    )

    assert len(sparade_resultat) == 1
    resultat = sparade_resultat[0].set_index("Modell")

    # Tre system jämförs
    assert len(sparade_resultat[0]) == 3

    champion = resultat.loc["SVR vanlig (T+6) (utan routing)"]
    routat = resultat.loc[
        "Routat system (SVR vanlig (T+6) + SVR nisch (T+6))"
    ]

    # 2 av 5 rader routas
    assert (resultat["routade rader"] == 2).all()

    # Routern är bevisligen av under gränsen: identisk MAE
    assert routat["MAE under gräns"] == champion["MAE under gräns"]
    assert champion["MAE under gräns"] == pytest.approx(0.5)

    # Total-MAE: champion 0,5 överallt; routat = (3*0,5 + 2*2)/5 = 1,1
    assert champion["MAE"] == pytest.approx(0.5)
    assert routat["MAE"] == pytest.approx(1.1)

    # High-PM-mått (y >= 5 -> raderna 10 och 11):
    # champion +0,5; nisch +2: routat är sämre här I DETTA exemplet —
    # siffrorna visar bara att systemen skiljer sig rätt.
    assert champion["High PM2.5 MAE"] == pytest.approx(0.5)
    assert routat["High PM2.5 MAE"] == pytest.approx(2.0)
    assert routat["High PM2.5 Count"] == 2


def test_jamfor_routat_system_hoppar_over_utan_gransrade_rader(tmp_path, monkeypatch):
    """Alla eller inga rader över gränsen -> jämförelsen hoppas över
    (rutningseffekten kan inte visas)."""
    class Dummy:
        def predict(self, X):
            return X["pm25"].to_numpy()

    val_df = pd.DataFrame({"pm25": [1, 2, 3], "target": [1, 2, 3]})
    val_path = tmp_path / "val.csv"
    val_df.to_csv(val_path, index=False)

    cfg = SimpleNamespace(
        features=["pm25"],
        target_column="target",
        jamfordelse_log=tmp_path / "jamforelse.csv",
    )

    sparade_resultat = []
    monkeypatch.setattr(
        "src.metrics.append_jamfordelse_log",
        lambda path, df: sparade_resultat.append(df.copy()),
    )

    # Alla rader under gränsen respektive alla rader över gränsen
    jamfor_routat_system(
        cfg=cfg, champion_modell=Dummy(), champion_namn="C",
        niche_modell=Dummy(), niche_namn="N",
        niche_threshold=100.0, val_path=val_path,
    )
    jamfor_routat_system(
        cfg=cfg, champion_modell=Dummy(), champion_namn="C",
        niche_modell=Dummy(), niche_namn="N",
        niche_threshold=0.5, val_path=val_path,
    )

    assert sparade_resultat == []


# ----------------------------------------------------------
# Kompositscore (plan 02)
# ----------------------------------------------------------

VIKTER = {
    "w_mae": 0.5,
    "w_high_mae": 0.35,
    "w_high_bias": 0.15,
    "min_antal_high": 10,
}


def test_composite_score_handraknad_fixture():
    """Kärnformeln: 0,5·3 + 0,35·5 + 0,15·|−2| = 3,55 (lägre = bättre)."""
    metrics = {
        "MAE": 3.0,
        "High PM2.5 MAE": 5.0,
        "High PM2.5 Bias": -2.0,
        "High PM2.5 Count": 50,
    }

    assert composite_score(metrics, VIKTER) == pytest.approx(3.55)


def test_composite_score_absolut_bias():
    """Biasen vägs med beloppet: +2 och −2 ger samma score."""
    metrics_pos = {
        "MAE": 3.0, "High PM2.5 MAE": 5.0,
        "High PM2.5 Bias": 2.0, "High PM2.5 Count": 50,
    }
    metrics_neg = {
        "MAE": 3.0, "High PM2.5 MAE": 5.0,
        "High PM2.5 Bias": -2.0, "High PM2.5 Count": 50,
    }

    assert composite_score(metrics_pos, VIKTER) == pytest.approx(
        composite_score(metrics_neg, VIKTER)
    )


def test_composite_score_nan_high_renormaliserar_till_mae(caplog):
    """NaN High-PM-mått (inga toppar i snittet) -> termerna droppas och
    vikterna renormaliseras: scoren blir ren MAE + varning loggas."""
    metrics = {
        "MAE": 3.0,
        "High PM2.5 MAE": float("nan"),
        "High PM2.5 Bias": float("nan"),
        "High PM2.5 Count": 0,
    }

    with caplog.at_level("WARNING", logger="src.metrics"):
        score = composite_score(metrics, VIKTER)

    assert score == pytest.approx(3.0)
    assert "High-PM-underlag" in caplog.text


def test_composite_score_fa_rader_over_gransen_fallback(caplog):
    """Färre än min_antal_high rader -> samma renormaliserade fallback."""
    metrics = {
        "MAE": 4.0,
        "High PM2.5 MAE": 9.0,
        "High PM2.5 Bias": -1.0,
        "High PM2.5 Count": 9,   # < min_antal_high = 10
    }

    with caplog.at_level("WARNING", logger="src.metrics"):
        score = composite_score(metrics, VIKTER)

    assert score == pytest.approx(4.0)
    assert "High-PM-underlag" in caplog.text


def test_composite_score_anpassad_min_grans():
    """min_antal_high är konfigurerbart: 9 rader räcker när kravet är 5."""
    metrics = {
        "MAE": 3.0,
        "High PM2.5 MAE": 5.0,
        "High PM2.5 Bias": -2.0,
        "High PM2.5 Count": 9,
    }

    assert composite_score(metrics, {**VIKTER, "min_antal_high": 5}) == (
        pytest.approx(3.55)
    )


def test_composite_score_kraver_mae():
    """Utan giltigt MAE finns inget att poängsätta -> ValueError."""
    with pytest.raises(ValueError):
        composite_score({"MAE": float("nan")}, VIKTER)
    with pytest.raises(ValueError):
        composite_score({}, VIKTER)


def test_metric_direction_composite_ar_lagre_battre():
    assert metric_direction("Validation Composite") == "lower"
    assert metric_direction("Composite") == "lower"


def test_select_best_model_composite():
    """Lägsta komposit vinner, baslnjen inkluderad."""
    results_df = pd.DataFrame([
        {"Modell": "SVR", "Validation Composite": 2.40},
        {"Modell": "Ridge Regression", "Validation Composite": 2.55},
    ])

    namn, rad = select_best_model(results_df, "Validation Composite")

    assert namn == "SVR"
    assert rad["Validation Composite"] == pytest.approx(2.40)


def test_build_comparison_table_baseline_fa_r_komposit():
    """Baslinjens komposit beräknas från dess egna mått (med samma
    vikter), så att även baslinjen kan vinna valet ärligt."""
    results_df = pd.DataFrame([
        {
            "Modell": "SVR",
            "Validation MAE": 3.0,
            "Validation High PM2.5 MAE": 5.0,
            "Validation High PM2.5 Bias": -2.0,
            "Validation High PM2.5 Count": 50,
            "Validation Composite": 3.55,
        },
    ])
    baseline_metrics = {
        "MAE": 4.0,
        "High PM2.5 MAE": 6.0,
        "High PM2.5 Bias": -3.0,
        "High PM2.5 Count": 50,
    }

    comparison = build_comparison_table(
        results_df,
        baseline_metrics,
        sort_by="Validation Composite",
        vikter=VIKTER,
    )

    assert comparison.iloc[0]["Modell"] == "SVR"

    baseline = comparison[
        comparison["Modell"] == "Persistence Baseline"
    ].iloc[0]

    # Handräkning: 0,5·4 + 0,35·6 + 0,15·3 = 2 + 2,1 + 0,45 = 4,55
    assert baseline["Validation Composite"] == pytest.approx(4.55)