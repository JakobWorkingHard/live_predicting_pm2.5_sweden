import numpy as np
import pandas as pd
import pytest

import src.champion
from src.champion import compete_with_champion


class DummyModel:
    def __init__(self, predictions):
        self.predictions = np.array(predictions)

    def predict(self, X):
        return self.predictions


# Samma vikter som [model.composite] i config.toml, men med ett lågt
# underlagskrav så att High-PM-termerna faktiskt används i testen.
VIKTER = {
    "w_mae": 0.5,
    "w_high_mae": 0.35,
    "w_high_bias": 0.15,
    "min_antal_high": 3,
}

FEATURES = ["pm25"]
TARGET = "target"


def _skriv_valdata(tmp_path, namn="val.csv"):
    """Valideringsdata: target = pm25 + 2, 30 rader (pm25 1..30)."""
    pm25 = list(range(1, 31))
    df = pd.DataFrame({
        "pm25": pm25,
        "target": [v + 2 for v in pm25],
    })
    path = tmp_path / namn
    df.to_csv(path, index=False)
    return path


def _perseverans(X):
    """Persistence-prediktion: pm25 består (felet blir -2 överallt)."""
    return X["pm25"].to_numpy()


def test_duell_strikt_battre_komposit_vinner(tmp_path):
    """Plan 02: strikt lägre kompositscore på valideringsdata vinner."""
    val_path = _skriv_valdata(tmp_path)
    val_df = pd.read_csv(val_path)

    # Ny modell: perfekt (komposit 0). Gammal: persistence (komposit > 0).
    ny = DummyModel(val_df["target"].to_numpy())
    gammal = DummyModel(_perseverans(val_df))

    winner, results_df = compete_with_champion(
        new_model=ny,
        new_name="Ny",
        old_model=gammal,
        old_name="Gammal",
        data_path=val_path,
        features=FEATURES,
        target=TARGET,
        threshold=20,
        vikter=VIKTER,
    )

    assert winner == "new"
    assert results_df.iloc[0]["Composite"] == pytest.approx(0.0)
    assert results_df.iloc[1]["Composite"] > 0


def test_duell_samma_score_behaller_tidigare_champion(tmp_path):
    """Tie-regeln (beslut 2026-10-03): lika komposit -> gamla championen."""
    val_path = _skriv_valdata(tmp_path)
    val_df = pd.read_csv(val_path)

    identisk_pred = _perseverans(val_df)

    winner, results_df = compete_with_champion(
        new_model=DummyModel(identisk_pred),
        new_name="Ny",
        old_model=DummyModel(identisk_pred),
        old_name="Gammal",
        data_path=val_path,
        features=FEATURES,
        target=TARGET,
        threshold=20,
        vikter=VIKTER,
    )

    assert winner == "old"
    assert results_df.iloc[0]["Composite"] == pytest.approx(
        results_df.iloc[1]["Composite"]
    )


def test_duell_samre_komposit_behaller_tidigare_champion(tmp_path):
    """Kandidat med högre komposit förlorar."""
    val_path = _skriv_valdata(tmp_path)
    val_df = pd.read_csv(val_path)

    winner, _ = compete_with_champion(
        new_model=DummyModel(_perseverans(val_df) + 5),
        new_name="Ny",
        old_model=DummyModel(_perseverans(val_df)),
        old_name="Gammal",
        data_path=val_path,
        features=FEATURES,
        target=TARGET,
        threshold=20,
        vikter=VIKTER,
    )

    assert winner == "old"


def test_duell_utan_tidigare_champion_vinner_ny(tmp_path):
    """Ingen tidigare champion -> kandidaten vinner automatiskt."""
    val_path = _skriv_valdata(tmp_path)
    val_df = pd.read_csv(val_path)

    winner, results_df = compete_with_champion(
        new_model=DummyModel(_perseverans(val_df)),
        new_name="Ny",
        old_model=None,
        old_name=None,
        data_path=val_path,
        features=FEATURES,
        target=TARGET,
        threshold=20,
        vikter=VIKTER,
    )

    assert winner == "new"
    assert len(results_df) == 1


def test_duell_laeser_valideringsfil_aldrig_testfil(tmp_path, monkeypatch):
    """Plan 02: duellen läser VALIDERINGSfilen — testfilen rör aldrig
    något val (assert på sökvägsargumentet)."""
    val_path = _skriv_valdata(tmp_path, "val.csv")
    # Lockrop: en testfil med annorlunda innehåll som ALDRIG får läsas.
    test_path = _skriv_valdata(tmp_path, "test.csv")
    val_df = pd.read_csv(val_path)

    lasta_anrop = []
    original = pd.read_csv

    def spion(path, *args, **kwargs):
        lasta_anrop.append(str(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(src.champion.pd, "read_csv", spion)

    winner, results_df = compete_with_champion(
        new_model=DummyModel(val_df["target"].to_numpy()),
        new_name="Ny",
        old_model=DummyModel(_perseverans(val_df)),
        old_name="Gammal",
        data_path=val_path,
        features=FEATURES,
        target=TARGET,
        threshold=20,
        vikter=VIKTER,
    )

    assert val_path.name in " ".join(lasta_anrop)
    assert all(test_path.name not in anrop for anrop in lasta_anrop)
    # Resultatet stämmer med VALIDERINGSfilens innehåll (perfekt modell)
    assert winner == "new"
    np.testing.assert_allclose(
        results_df.iloc[0]["MAE"], 0.0, atol=1e-9
    )


def test_duell_anvander_threshold_och_redovisar_komposit(tmp_path):
    """Gränsen styr High-PM-måtten i duelltabellen, och kompositen
    byggs av samma termer som turneringen."""
    val_path = _skriv_valdata(tmp_path)
    val_df = pd.read_csv(val_path)

    _, results_df = compete_with_champion(
        new_model=DummyModel(_perseverans(val_df)),
        new_name="Ny",
        old_model=DummyModel(_perseverans(val_df)),
        old_name="Gammal",
        data_path=val_path,
        features=FEATURES,
        target=TARGET,
        threshold=20,
        vikter=VIKTER,
    )

    for col in ("High PM2.5 MAE", "High PM2.5 Bias", "High PM2.5 Count", "Composite"):
        assert col in results_df.columns

    # Utan threshold finns inga High-PM-mått, och kompositen degraderar
    # till ren MAE (med varning, se composite_score).
    _, results_df_utan = compete_with_champion(
        new_model=DummyModel(_perseverans(val_df)),
        new_name="Ny",
        old_model=DummyModel(_perseverans(val_df)),
        old_name="Gammal",
        data_path=val_path,
        features=FEATURES,
        target=TARGET,
        vikter=VIKTER,
    )

    assert "High PM2.5 MAE" not in results_df_utan.columns
    assert results_df_utan.iloc[0]["Composite"] == pytest.approx(
        results_df_utan.iloc[0]["MAE"]
    )
