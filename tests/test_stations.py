"""
Tester för stationpars-modulen (src/stations.py) och konfigfabriken
(src/config.bygg_station_config).

Täcker besluten 2026-11-15:
  * PM2.5-station är huvudstation; närmaste väderstation inom max_km väljs
  * en väderstation FÅR återanvändas av flera PM2.5-stationer
  * PM2.5-serien måste aktivt rapportera (max_serie_alder_dagar)
  * minst min_traningsmanader månader träning efter valideringen
  * väderstationen måste täcka hela träningsfönstret
  * cache: färsk cache används, gamla cachen är fallback vid API-fel
  * per-par-konfigurationer namnryms (loggar, modeller, arkiv, filnamn)
  * rullande träningsstart: max(seriens start, idag - train_rullande_ar)
"""

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from dataclasses import fields as dc_fields

import pandas as pd
import pytest

import src.stations as st
from src.stations import (
    Pm25Station,
    StationPar,
    Vaderstation,
    haversine_km,
    hamta_pm25_stationer,
    hamta_stationpar,
    las_stationpar_cache,
    para_stationer,
    skriv_stationpar_cache,
)
from src.config import bygg_station_config, ladda_baskonfig

PROJEKT_ROT = Path(__file__).resolve().parents[1]

# Koordinater från det verkliga Malmö-paret (sanity: ~4,3 km)
DALAPLAN_LAT, DALAPLAN_LON = 55.584747, 13.006230
MALMO_A_LAT, MALMO_A_LON = 55.5715, 13.0708


# ----------------------------------------------------------------
# Fixturer
# ----------------------------------------------------------------

@pytest.fixture
def bas():
    """Riktig BasConfig ur repots config.toml (inga API-anrop sker)."""
    return ladda_baskonfig(PROJEKT_ROT / "config.toml", PROJEKT_ROT)


@pytest.fixture
def pm_dalaplan():
    return Pm25Station(
        station_id=272, namn="Malmö Dalaplan", klass="Gaturum",
        lat=DALAPLAN_LAT, lon=DALAPLAN_LON, tidsserie_id=6247,
        serie_start="2018-01-01", serie_slut="2026-11-15",
    )


@pytest.fixture
def vader_malmo_a():
    return Vaderstation(
        station_id=52350, namn="Malmö A",
        lat=MALMO_A_LAT, lon=MALMO_A_LON,
        data_fran=pd.Timestamp("1990-01-01", tz="Europe/Stockholm"),
        aktiv=True,
    )


def _ilskna_dagar_sedan(dagar: int):
    ms = (datetime.now(timezone.utc).timestamp() - dagar * 24 * 3600) * 1000
    return int(ms)


def _par(**over):
    """Standard-StationPar (Malmö) med valfria överskrivningar."""
    varden = dict(
        pm25_station_id=272, pm25_namn="Malmö Dalaplan", pm25_klass="Gaturum",
        pm25_lat=DALAPLAN_LAT, pm25_lon=DALAPLAN_LON,
        tidsserie_id=6247, serie_start="2018-01-01", serie_slut="2026-11-15",
        vaderstation_id=52350, vaderstation_namn="Malmö A",
        vaderstation_lat=MALMO_A_LAT, vaderstation_lon=MALMO_A_LON,
        avstand_km=4.315,
    )
    varden.update(over)
    return StationPar(**varden)


# ----------------------------------------------------------------
# haversine
# ----------------------------------------------------------------

def test_haversine_malmo_dalaplan_till_malmo_a():
    """Sanity: det verifierade Malmö-paret ligger ~4,3 km från varandra."""
    d = haversine_km(DALAPLAN_LAT, DALAPLAN_LON, MALMO_A_LAT, MALMO_A_LON)
    assert 4.0 < d < 4.6


def test_haversine_noll_avstand():
    assert haversine_km(55.5, 13.0, 55.5, 13.0) == pytest.approx(0.0)


# ----------------------------------------------------------------
# para_stationer
# ----------------------------------------------------------------

def test_para_stationer_valjer_narmaste(pm_dalaplan, vader_malmo_a):
    """Två kvalificerade väderstationer: den närmaste väljs."""
    # ~12,5 km bort från Dalaplan — kvalificerad men inte närmast
    lund = Vaderstation(
        station_id=53430, namn="Lund A", lat=55.7033, lon=13.1942,
        data_fran=pd.Timestamp("1995-01-01", tz="Europe/Stockholm"), aktiv=True,
    )
    par = para_stationer(
        [pm_dalaplan], [lund, vader_malmo_a],
        max_km=15, min_traningsmanader=24, validering_manader=6,
        train_rullande_ar=4,
    )
    assert len(par) == 1
    assert par[0].vaderstation_id == 52350
    assert par[0].vaderstation_namn == "Malmö A"
    assert par[0].avstand_km < 5.0


def test_vaderstation_ateranvands(pm_dalaplan, vader_malmo_a):
    """Två PM2.5-stationer FÅR dela på samma närmaste väderstation."""
    radhuset = Pm25Station(
        station_id=290, namn="Malmö Rådhuset", klass="Urban bakgrund",
        lat=55.6050, lon=13.0000, tidsserie_id=6161,
        serie_start="2018-01-01", serie_slut="2026-11-15",
    )
    par = para_stationer(
        [pm_dalaplan, radhuset], [vader_malmo_a],
        max_km=10, min_traningsmanader=24, validering_manader=6,
        train_rullande_ar=4,
    )
    assert len(par) == 2
    assert {p.vaderstation_id for p in par} == {52350}
    assert {p.pm25_namn for p in par} == {"Malmö Dalaplan", "Malmö Rådhuset"}


def test_para_utanfor_radie_hoppas_over(pm_dalaplan, vader_malmo_a):
    """Ingen väderstation inom max_km -> inget par."""
    avlagsen = Vaderstation(
        station_id=99999, namn="Kiruna A", lat=67.8556, lon=20.2253,
        data_fran=pd.Timestamp("1990-01-01", tz="Europe/Stockholm"), aktiv=True,
    )
    par = para_stationer(
        [pm_dalaplan], [avlagsen],
        max_km=10, min_traningsmanader=24, validering_manader=6,
        train_rullande_ar=4,
    )
    assert par == []


def test_min_traningsmanader_hoppas_over(vader_malmo_a):
    """Serien är yngre än min_traningsmanader + validering -> inget par,
    även om väderstationen finns nära."""
    ung = Pm25Station(
        station_id=1, namn="Ung station", klass="Gaturum",
        lat=DALAPLAN_LAT, lon=DALAPLAN_LON, tidsserie_id=111,
        serie_start="2026-01-01", serie_slut="2026-11-15",
    )
    par = para_stationer(
        [ung], [vader_malmo_a],
        max_km=10, min_traningsmanader=24, validering_manader=6,
        train_rullande_ar=4,
    )
    assert par == []


def test_valderstation_maste_tacka_traningsfonstret(pm_dalaplan):
    """Närmaste väderstation saknar data för delar av fönstret ->
    den täckande men avlägsnare stationen väljs istället."""
    # Dalaplans serie börjar 2018 men fönstret kapas till idag - 4 år,
    # dvs ~2022 — väderstationen med data_fran 2024 täcker inte fönstret.
    sen_vader = Vaderstation(
        station_id=1, namn="Sen väderstation", lat=MALMO_A_LAT, lon=MALMO_A_LON,
        data_fran=pd.Timestamp("2024-06-01", tz="Europe/Stockholm"), aktiv=True,
    )
    tidig_vader = Vaderstation(
        station_id=2, namn="Tidig väderstation",
        lat=MALMO_A_LAT + 0.05, lon=MALMO_A_LON + 0.05,  # ~7 km — längre bort
        data_fran=pd.Timestamp("1990-01-01", tz="Europe/Stockholm"), aktiv=True,
    )
    par = para_stationer(
        [pm_dalaplan], [sen_vader, tidig_vader],
        max_km=10, min_traningsmanader=24, validering_manader=6,
        train_rullande_ar=4,
    )
    assert len(par) == 1
    assert par[0].vaderstation_id == 2  # den täckande, inte den närmaste


# ----------------------------------------------------------------
# hamta_pm25_stationer (aktiva-serien-filtret)
# ----------------------------------------------------------------

def test_hamta_pm25_stationer_filtrerar_inaktiva_serier(monkeypatch):
    """
    Stationen väljer serien med senaste lastValue och hoppas över om den
    inte rapporterat inom max_serie_alder_dagar.
    """
    bulk = [
        # Aktiv station med en gammal död serie (126) och en aktuell (6247)
        {"id": "126", "label": "PM2.5 SPP-SE0058A_06001_100_100, Malmö Dalaplan, ...",
         "station": {"id": "272", "properties": {"label": "Malmö Dalaplan",
                     "classification": "Gaturum"},
                     "geometry": {"type": "Point", "coordinates": [13.0062, 55.5847]}}},
        {"id": "6247", "label": "PM2.5 SPP-SE0058A_06001_100_101, Malmö Dalaplan, ...",
         "station": {"id": "272", "properties": {"label": "Malmö Dalaplan",
                     "classification": "Gaturum"},
                     "geometry": {"type": "Point", "coordinates": [13.0062, 55.5847]}}},
        # Station som slutade rapportera för 40 dagar sedan
        {"id": "999", "label": "PM2.5 SPP-SEXXXX_06001_100_101, Staden, ...",
         "station": {"id": "333", "properties": {"label": "Igårstation",
                     "classification": "Urban bakgrund"},
                     "geometry": {"type": "Point", "coordinates": [15.0, 60.0]}}},
        # Icke-PM2.5-serie ska filtreras bort redan i bulk
        {"id": "111", "label": "NO2 SPP-SEXXXX_00008_100_101, Staden, ...",
         "station": {"id": "333", "properties": {"label": "NO2-station",
                     "classification": "Gaturum"},
                     "geometry": {"type": "Point", "coordinates": [15.0, 60.0]}}},
    ]

    def fake_api_get(url, forsok=4, timeout=(15, 60)):
        return SimpleNamespace(json=lambda: bulk)

    meta = {
        "126": {"first": _ilskna_dagar_sedan(2900), "last": _ilskna_dagar_sedan(700),
                "primary": "hour", "fenomen": "6001"},
        "999": {"first": _ilskna_dagar_sedan(2900), "last": _ilskna_dagar_sedan(40),
                "primary": "hour", "fenomen": "6001"},
    }

    def fake_meta(luft_api, tid):
        if tid == "6247":
            return {"first": _ilskna_dagar_sedan(1270), "last": _ilskna_dagar_sedan(0),
                    "primary": "minute", "fenomen": "6001"}
        return meta.get(tid)

    monkeypatch.setattr(st, "_api_get", fake_api_get)
    monkeypatch.setattr(st, "_hamta_tidsserie_meta", fake_meta)

    stationer = hamta_pm25_stationer("http://fake", max_serie_alder_dagar=14)

    # Bara Dalaplan klarar aktivitetsfiltret; NO2 serien räknades aldrig,
    # Igårstationen hoppas över (40 dagar > 14).
    assert [s.namn for s in stationer] == ["Malmö Dalaplan"]
    dalaplan = stationer[0]
    # Serien med SENASTE lastValue väljs (6247, inte den döda 126)
    assert dalaplan.tidsserie_id == 6247


# ----------------------------------------------------------------
# Cache
# ----------------------------------------------------------------

def test_cache_roundtrip(tmp_path):
    """Skriv + läs: StationPar överlever CSV-cachen oförändrat."""
    par = [_par(), _par(pm25_namn="Uppsala Kungsgatan 67", pm25_station_id=184,
                       tidsserie_id=4352, vaderstation_id=97510,
                       vaderstation_namn="Uppsala Aut", avstand_km=1.42)]
    cachefil = tmp_path / "stationspar_v1.csv"
    skriv_stationpar_cache(cachefil, par)
    lasta, beraknad = las_stationpar_cache(cachefil)
    assert beraknad is not None
    assert len(lasta) == 2
    for skrivet, last in zip(par, lasta):
        assert skrivet == last


def test_hamta_stationpar_anvander_farsk_cache(monkeypatch, tmp_path):
    """Färsk cache -> ingen API-upptäckt alls."""
    bas_min = SimpleNamespace(
        projekt_rot=tmp_path,
        stationpar_cache="stationspar_v1.csv",
        cache_giltig_dagar=7,
        luft_api="http://fake", vader_api="http://fake",
        vader_parametrar={"Lufttemperatur": 1},
        max_km=10, max_serie_alder_dagar=14, min_traningsmanader=24,
        validering_manader=6, train_rullande_ar=4,
    )
    skriv_stationpar_cache(tmp_path / bas_min.stationpar_cache, [_par()])

    def boom(*a, **k):
        # KeyboardInterrupt fångas INTE av hamta_stationpars except Exception —
        # om upptäckten ändå körs misslyckas testet högljudd.
        raise KeyboardInterrupt("Upptäcken fick inte köras — cachen är färsk.")

    monkeypatch.setattr(st, "hamta_pm25_stationer", boom)
    par = hamta_stationpar(bas_min)
    assert [p.pm25_namn for p in par] == ["Malmö Dalaplan"]


def test_hamta_stationpar_faller_tillbaka_pa_gammal_cache(monkeypatch, tmp_path):
    """Upptäckt misslyckas (API nere) -> gammal cache används ändå."""
    bas_min = SimpleNamespace(
        projekt_rot=tmp_path,
        stationpar_cache="stationspar_v1.csv",
        cache_giltig_dagar=7,
        luft_api="http://fake", vader_api="http://fake",
        vader_parametrar={"Lufttemperatur": 1},
        max_km=10, max_serie_alder_dagar=14, min_traningsmanader=24,
        validering_manader=6, train_rullande_ar=4,
    )
    skriv_stationpar_cache(tmp_path / bas_min.stationpar_cache, [_par()])

    # Föråldra cachen så upptäckten faktiskt förs
    cachefil = tmp_path / bas_min.stationpar_cache
    df = pd.read_csv(cachefil)
    df["beraknad_tid"] = "2020-01-01T00:00:00"
    df.to_csv(cachefil, index=False)

    def dalig_upptackt(*a, **k):
        raise ConnectionError("SMHI nere")

    monkeypatch.setattr(st, "hamta_pm25_stationer", dalig_upptackt)
    par = hamta_stationpar(bas_min)
    assert [p.pm25_namn for p in par] == ["Malmö Dalaplan"]


# ----------------------------------------------------------------
# bygg_station_config (namnrymdning + rullande träningsstart)
# ----------------------------------------------------------------

@pytest.fixture
def malmo_par():
    return _par(serie_start="2023-05-26")


@pytest.fixture
def uppsala_par():
    return _par(
        pm25_station_id=184, pm25_namn="Uppsala Kungsgatan 67",
        pm25_klass="Gaturum", pm25_lat=59.8583, pm25_lon=17.6389,
        tidsserie_id=4352, serie_start="2017-02-02",
        vaderstation_id=97510, vaderstation_namn="Uppsala Aut",
        vaderstation_lat=59.8583, vaderstation_lon=17.6460,
        avstand_km=1.42,
    )


def _bygg_med_fast_arkivtid(monkeypatch, bas_cfg, par):
    """Factory med fast arkivtidpunkt så datumen blir deterministiska."""
    monkeypatch.setattr(
        "src.extracting_data.hamta_senaste_arkivtid",
        lambda **k: pd.Timestamp("2026-11-01 12:00", tz="Europe/Stockholm"),
    )
    return bygg_station_config(bas_cfg, par)


def test_bygg_station_config_skiljer_alla_vagar_at(monkeypatch, bas, malmo_par, uppsala_par):
    """Två par får ALDRIG samma väg i någon namnrymd."""
    # Baskatalogerna delas medvetet av alla par — allt ANNAT som är en
    # sökväg måste vara unikt per par.
    delade_kataloger = {"raw_dir", "merged_dir", "validated_dir", "processed_dir"}

    def _path_falt(cfg):
        return {
            f.name: getattr(cfg, f.name)
            for f in dc_fields(cfg)
            if isinstance(getattr(cfg, f.name), Path)
        }

    vm = _path_falt(_bygg_med_fast_arkivtid(monkeypatch, bas, malmo_par))
    vu = _path_falt(_bygg_med_fast_arkivtid(monkeypatch, bas, uppsala_par))

    for falt in set(vm) & set(vu) - delade_kataloger:
        assert vm[falt] != vu[falt], f"{falt} krockar mellan par"

    # ... och tidsserie-ID:t ska synas i logg- och arkivnamnen
    assert "6247" in vm["champion_log"].name
    assert "4352" in vu["champion_log"].name
    assert vm["models_dir"].name == "6247"
    assert vu["models_dir"].name == "4352"
    assert vm["prognos_dir"] != vu["prognos_dir"]
    assert vm["prediktioner_log"] != vu["prediktioner_log"]
    assert vm["test_last_dir"] != vu["test_last_dir"]


def test_bygg_station_config_rullande_traningsstart(monkeypatch, bas, malmo_par, uppsala_par):
    """
    train_start = max(seriens start, idag - train_rullande_ar år):
      * Uppsala (serie 2017, äldre än fönstret) -> idag - 4 år
      * Malmö (serie 2023-05, yngre än fönstret) -> seriens start
    """
    cfg_uppsala = _bygg_med_fast_arkivtid(monkeypatch, bas, uppsala_par)
    cfg_malmo = _bygg_med_fast_arkivtid(monkeypatch, bas, malmo_par)

    rullande = (
        pd.Timestamp.now(tz="Europe/Stockholm").normalize()
        - pd.DateOffset(years=bas.train_rullande_ar)
    ).strftime("%Y-%m-%d")

    assert cfg_uppsala.train_start == rullande
    assert cfg_malmo.train_start == "2023-05-26"


def test_bygg_station_config_dynamiska_perioder(monkeypatch, bas, malmo_par):
    """Fast arkivtid 2026-11-01 -> validering = maj-okt 2026, train slut 30 apr."""
    cfg = _bygg_med_fast_arkivtid(monkeypatch, bas, malmo_par)
    assert cfg.validering_slut == "2026-10-31"
    assert cfg.validering_start == "2026-05-01"
    assert cfg.train_slut == "2026-04-30"
    assert cfg.train_start == "2023-05-26"  # serien är yngre än 4-årsfönstret


def test_bygg_station_config_prognos_url_per_par(monkeypatch, bas, malmo_par, uppsala_par):
    """Prognos-URL:n formateras med parets egna väderstationskoordinater."""
    cfg_malmo = _bygg_med_fast_arkivtid(monkeypatch, bas, malmo_par)
    assert f"{malmo_par.vaderstation_lon}" in cfg_malmo.prognos_url
    assert f"{malmo_par.vaderstation_lat}" in cfg_malmo.prognos_url

    cfg_uppsala = _bygg_med_fast_arkivtid(monkeypatch, bas, uppsala_par)
    assert str(uppsala_par.vaderstation_lon) in cfg_uppsala.prognos_url
    assert cfg_uppsala.prognos_url != cfg_malmo.prognos_url
