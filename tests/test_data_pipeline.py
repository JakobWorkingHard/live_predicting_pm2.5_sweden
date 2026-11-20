"""Tester för det låsta test-snapshottet (plan 02): data/test/last
fryser en testperiod för oberoende RAPPORTERING — aldrig modellval."""

import json

import pandas as pd
import pytest

import src.data_pipeline as dp
from src.data_pipeline import las_last_test_vagar, skapa_last_test_snapshot


_LASTNAMN = {
    "manifest": "manifest.json",
    "test_processed": "test_processed.csv",
    "test_processed_forskjuten": "test_processed_forskjuten.csv",
    "test_merged": "test_merged.csv",
}


def _cfg(tmp_path, **over):
    class _Cfg:
        pass

    cfg = _Cfg()
    cfg.test_last_dir = tmp_path / "test_last"
    cfg.pm25_station = "Malmö Dalaplan"
    cfg.pm25_tidsserie_id = 6247
    cfg.vaderstation = "Malmö A"
    cfg.vaderstation_id = 52350
    cfg.horisont = 6
    for namn, varde in over.items():
        setattr(cfg, namn, varde)
    return cfg


def _komplett_snapshot(last_dir):
    (last_dir / _LASTNAMN["manifest"]).write_text("{}", encoding="utf-8")
    for namn in ("test_processed", "test_processed_forskjuten"):
        pd.DataFrame({"pm25": [1]}).to_csv(
            last_dir / _LASTNAMN[namn], index=False,
        )


def test_las_last_test_vagar_saknas_returnerar_none(tmp_path):
    cfg = _cfg(tmp_path)
    assert las_last_test_vagar(cfg) is None


def test_las_last_test_vagar_ofullstandigt_returnerar_none(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.test_last_dir.mkdir(parents=True)
    (cfg.test_last_dir / _LASTNAMN["manifest"]).write_text(
        "{}", encoding="utf-8",
    )
    # Endast manifestet finns — de processade filerna saknas.
    assert las_last_test_vagar(cfg) is None


def test_las_last_test_vagar_komplett_returnerar_sokvagar(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.test_last_dir.mkdir(parents=True)
    _komplett_snapshot(cfg.test_last_dir)

    vagor = las_last_test_vagar(cfg)

    assert vagor is not None
    assert vagor["test_processed"].is_file()
    assert vagor["test_processed_forskjuten"].is_file()
    assert vagor["manifest"].is_file()


def test_skapa_last_test_snapshot_vagrar_vid_befintligt(tmp_path, monkeypatch):
    """Ett befintligt snapshot är ett versionsbeslut: utan --ersatt
    hämtas ingen ny data (API-anropet får aldrig ske)."""
    cfg = _cfg(tmp_path)
    cfg.test_last_dir.mkdir(parents=True)
    _komplett_snapshot(cfg.test_last_dir)

    def fetch_skall_inte_kora(cfg):
        raise AssertionError("fetch_and_merge_test fick inte anropas")

    monkeypatch.setattr(dp, "fetch_and_merge_test", fetch_skall_inte_kora)

    assert skapa_last_test_snapshot(cfg, ersatt=False) is False


def test_skapa_last_test_snapshot_med_ersatt(tmp_path, monkeypatch):
    """Med --ersatt skrivs ett nytt snapshot: källfilen kopieras, de två
    processade varianterna byggs och manifestet noterar period, stationer
    och att källan är preliminär latest-months-data."""
    cfg = _cfg(tmp_path)

    # Källan som fetch_and_merge_test "hämtar": en sammanfogad fil.
    merged_kalla = tmp_path / "test_merged_kalla.csv"
    pd.DataFrame({"pm25": [1, 2], "target": [3, 4]}).to_csv(
        merged_kalla, index=False,
    )

    def fake_fetch(cfg):
        return {
            "test_merged": merged_kalla,
            "test_start": "2026-07-01 00:00",
            "test_slut": "2026-11-14 20:00",
        }

    monkeypatch.setattr(dp, "fetch_and_merge_test", fake_fetch)

    byggda = []
    monkeypatch.setattr(
        dp,
        "build_processed_dataset",
        lambda cfg, df, output_path, mal_vader: (
            byggda.append((str(output_path), mal_vader)),
            pd.DataFrame({"pm25": [1]}).to_csv(output_path, index=False),
        ),
    )

    assert skapa_last_test_snapshot(cfg, ersatt=False) is True

    last_dir = cfg.test_last_dir
    assert (last_dir / _LASTNAMN["test_processed"]).is_file()
    assert (last_dir / _LASTNAMN["test_processed_forskjuten"]).is_file()
    assert (last_dir / _LASTNAMN["test_merged"]).is_file()

    # Båda varianterna byggs ur SAMMA sammanfogade data; vanligt väder
    # (False) före måltidsvädret (True) i byggordningen spelar ingen
    # roll, men båda må ha byggts.
    assert sorted(mal_vader for _, mal_vader in byggda) == [False, True]

    manifest = json.loads(
        (last_dir / _LASTNAMN["manifest"]).read_text(encoding="utf-8"),
    )

    assert manifest["pm25_tidsserie_id"] == 6247
    assert manifest["vaderstation_id"] == 52350
    assert manifest["test_period"]["start"] == "2026-07-01 00:00"
    assert manifest["test_period"]["slut"] == "2026-11-14 20:00"
    assert "latest-months" in manifest["kalla"]
    assert "ALDRIG modellval" in manifest["anvandning"]

    # Las_last_test_vagar hittar nu det kompletta snapshotet.
    assert las_last_test_vagar(cfg) is not None

    # Om-snapshot utan --ersatt vägrar nu igen.
    assert skapa_last_test_snapshot(cfg, ersatt=False) is False
