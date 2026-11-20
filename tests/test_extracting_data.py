"""Tester för src/extracting_data.py."""

import re

import pandas as pd
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.extracting_data import hamta_senaste_pm25, stada_prognos_dir


def _cfg(tmp_path):
    """Minimal config med bara de fält som hamta_senaste_pm25 och
    stada_prognos_dir använder."""
    return SimpleNamespace(
        prognos_dir=tmp_path / "prognoser",
        pm25_tidsserie_id=123,
        luft_api="https://example.test/luft",
        prognos_bibehall_dagar=30,
        pm25_arkiv_bibehall_dagar=7,
    )


def _fake_pm25(n=3):
    """Liten pm25-dataframe i samma format som hamta_pm25 returnerar."""
    tid = pd.date_range(pd.Timestamp.now(tz="Europe/Stockholm"), periods=n, freq="h")
    return pd.DataFrame({"pm25": [5.0, 6.0, 7.0][:n]}, index=tid).rename_axis("tid")


def test_hamta_senaste_pm25_fyller_arkiv_och_tvingar_fram_ny_data(tmp_path):
    """Wrapperns två verkliga beteenden:

    1) csv_path får INTE finnas när hamta_pm25 anropas — annars returnerar
       den cachad data och live-prediktionen bygger på gamla mätvärden.
    2) Hämtningen sparas i prognos_dir/pm25/ (hamta_pm25 skriver aldrig
       csv_path själv, så arkivet fylls annars aldrig).
    """

    def fake_hamta_pm25(start, slut, tidsserie_id, luft_api, csv_path):
        # Härma riktiga hamta_pm25:s första rad: finns filen -> returnera cache.
        assert not Path(str(csv_path)).is_file(), (
            f"csv_path {csv_path} fanns redan — den timstämplade sökvägen "
            "måste vara unik för att tvinga fram ny data"
        )
        return _fake_pm25()

    cfg = _cfg(tmp_path)

    with patch("src.extracting_data.hamta_pm25", side_effect=fake_hamta_pm25):
        hamta_senaste_pm25(cfg)

    # Arkivet fylls: exakt en sparad hämtning på rätt ställe
    sparade = list((cfg.prognos_dir / "pm25").glob("pm25_*.csv"))
    assert len(sparade) == 1


def test_hamta_senaste_pm25_en_fil_per_heltimme(tmp_path):
    """Sökvägen är timstämplad (H00), inte sekundstämplad: upprepade
    körningar inom samma timme återanvänder timmens hämtning och arkivet
    hålls avdubblat — det växer annars med en fil PER KÖRNING i evighet."""
    anrop = {"antal": 0}

    def fake_hamta_pm25(start, slut, tidsserie_id, luft_api, csv_path):
        anrop["antal"] += 1
        # Härma cachen: finns filen -> returnera den befintliga datan.
        if Path(str(csv_path)).is_file():
            return pd.read_csv(csv_path, index_col="tid", parse_dates=True)
        return _fake_pm25()

    cfg = _cfg(tmp_path)

    with patch("src.extracting_data.hamta_pm25", side_effect=fake_hamta_pm25):
        hamta_senaste_pm25(cfg)
        hamta_senaste_pm25(cfg)

    sparade = list((cfg.prognos_dir / "pm25").glob("pm25_*.csv"))
    assert len(sparade) == 1
    assert anrop["antal"] == 2  # hamta_pm25 anropas, men cachen skyddar
    # Filnamnet är på heltimmesformat, t.ex. pm25_20261115T1400.csv
    assert re.fullmatch(r"pm25_\d{8}T\d{2}00\.csv", sparade[0].name)


def _skapa_fil(katalog, namn):
    katalog.mkdir(parents=True, exist_ok=True)
    (katalog / namn).write_text("arkivdata", encoding="utf-8")


def test_stada_prognos_dir_tar_bara_gamla_arkivfiler(tmp_path):
    """Endast projektets egna arkivfiler äldre än bibehåll-fönstren tas
    bort. Färska filer och främmande filnamn lämnas orörda."""
    cfg = _cfg(tmp_path)
    nu_utc = pd.Timestamp.now(tz="UTC")
    nu_svensk = pd.Timestamp.now(tz="Europe/Stockholm")

    # Prognosarkiv: 40 dagar gammal (tas bort) + färska (behålls)
    _skapa_fil(cfg.prognos_dir, f"prognos_{nu_utc - pd.Timedelta(days=40):%Y-%m-%dT%H%M}_utc.json")
    _skapa_fil(cfg.prognos_dir, f"prognos_{nu_utc:%Y-%m-%dT%H%M}_utc.json")
    _skapa_fil(cfg.prognos_dir, f"prognos_{nu_utc:%Y-%m-%dT%H%M}_utc_{24}h.csv")
    # pm25-arkiv: 10 dagar gammal (tas bort) + färska (behålls)
    _skapa_fil(cfg.prognos_dir / "pm25", f"pm25_{nu_svensk - pd.Timedelta(days=10):%Y%m%dT%H00}.csv")
    _skapa_fil(cfg.prognos_dir / "pm25", f"pm25_{nu_svensk:%Y%m%dT%H00}.csv")
    # Främmande filer ska aldrig röras
    _skapa_fil(cfg.prognos_dir / "pm25", "oviktig_fil.csv")
    _skapa_fil(cfg.prognos_dir, "prognos_inte_en_klockslag_utc.json")

    antal = stada_prognos_dir(cfg)

    assert antal == 2
    assert not (cfg.prognos_dir / f"prognos_{nu_utc - pd.Timedelta(days=40):%Y-%m-%dT%H%M}_utc.json").exists()
    assert (cfg.prognos_dir / f"prognos_{nu_utc:%Y-%m-%dT%H%M}_utc.json").exists()
    assert (cfg.prognos_dir / "prognos_inte_en_klockslag_utc.json").exists()
    assert not (cfg.prognos_dir / "pm25" / f"pm25_{nu_svensk - pd.Timedelta(days=10):%Y%m%dT%H00}.csv").exists()
    assert (cfg.prognos_dir / "pm25" / f"pm25_{nu_svensk:%Y%m%dT%H00}.csv").exists()
    assert (cfg.prognos_dir / "pm25" / "oviktig_fil.csv").exists()


def test_stada_prognos_dir_hanterar_gammalt_pm25_format(tmp_path):
    """pm25-filer sparade med det äldre sekundformatet (före
    heltimmes-avdubblingen) städas bort av samma regel."""
    cfg = _cfg(tmp_path)
    # Sekundformat: pm25_<YYYYMMDDTHHMMSS>.csv, klart äldre än 7 dagar
    _skapa_fil(cfg.prognos_dir / "pm25", "pm25_20261023T140904.csv")

    antal = stada_prognos_dir(cfg)

    assert antal == 1
    assert not (cfg.prognos_dir / "pm25" / "pm25_20261023T140904.csv").exists()


def test_stada_prognos_dir_saknad_katalog(tmp_path):
    """Ingen prognosmapp = inget att städa, inga krascher."""
    cfg = _cfg(tmp_path)
    assert stada_prognos_dir(cfg) == 0
