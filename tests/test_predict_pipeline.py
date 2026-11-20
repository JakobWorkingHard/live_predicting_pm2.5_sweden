"""Tester för src/predict_pipeline.py (live-prediktioner), inklusive
nisch-routern för höga PM2.5-halter (plan 01)."""

import pandas as pd
import pytest

from types import SimpleNamespace

from src.io import save_champion_model, append_champion_log
import src.predict_pipeline as pp
from src.predict_pipeline import (
    bestam_ankarpunkt,
    visa_senaste_prediktioner,
    bestam_vanlig_champion,
    hamta_nisch_champion,
    kor_live_prediktion,
    visa_skip_sammanstallning,
)
from src.metrics import sammanstall_live_uppfoljning


def _tim(n, start="2026-10-21 10:00"):
    """Heltimmesindex i svensk tid, samma format som live-hämtningarna."""
    return pd.date_range(start, periods=n, freq="h", tz="Europe/Stockholm")


def test_bestam_ankarpunkt_valjer_senaste_kompletta_timme():
    """T_obs = senaste timme med pm25 och alla vaderparametrar kompletta."""
    tid = _tim(5)  # 10:00–14:00
    pm25 = pd.DataFrame({"pm25": [8.0, 8.1, 8.2, 8.3, 8.4]}, index=tid)
    vader = pd.DataFrame(
        {
            "Lufttemperatur": [5.0, 5.1, None, 5.3, 5.4],  # 12:00 ofullständig
            "Vindriktning": [10.0, 20.0, 30.0, 40.0, 360.0],
            "Vindhastighet": [1.0, 1.1, 1.2, 1.3, 1.4],
            "Relativ luftfuktighet": [70.0, 71.0, 72.0, 73.0, 74.0],
            "Nederbördsmängd": [0.0, 0.1, 0.2, 0.3, 0.4],
        },
        index=tid,
    )

    T_obs = bestam_ankarpunkt(pm25, vader)

    # Senaste KOMPLETTA timmen är 14:00 (12:00 saknar Lufttemperatur).
    # 360° i Vindriktning hindrar inte: ankarpunkten letar kompletthet,
    # inte validitet — 360 -> 0 är sanity_kolla_vaders jobb senare i flödet.
    assert T_obs == pd.Timestamp("2026-10-21 14:00", tz="Europe/Stockholm")


def test_bestam_ankarpunkt_faller_tillbaka_nar_senaste_ar_inkomplett():
    """Är den senaste timmen ofullständig väljs näst senaste kompletta."""
    tid = _tim(3)  # 10:00–12:00
    pm25 = pd.DataFrame({"pm25": [8.0, 8.1, 8.2]}, index=tid)
    vader = pd.DataFrame({"Lufttemperatur": [5.0, 5.1, None]}, index=tid)

    T_obs = bestam_ankarpunkt(pm25, vader)

    assert T_obs == pd.Timestamp("2026-10-21 11:00", tz="Europe/Stockholm")


def test_bestam_ankarpunkt_returnerar_none_om_inget_komplett():
    """Ingen komplett timme -> None (kor_live_prediktion hoppar senare över)."""
    tid = _tim(2)
    # 10:00 saknar pm25, 11:00 saknar Lufttemperatur -> ingen komplett timme
    pm25 = pd.DataFrame({"pm25": [None, 8.2]}, index=tid)
    vader = pd.DataFrame({"Lufttemperatur": [5.0, None]}, index=tid)

    assert bestam_ankarpunkt(pm25, vader) is None


def test_bestam_ankarpunkt_tolererar_duplikata_tidsstampelar():
    """Regression: duplikat i INGÅNGEN måste bort FÖRE concat — annars
    kraschar sammanfogningen ('cannot reindex from a duplicate axis')."""
    tid = _tim(2)
    pm25 = pd.DataFrame(
        {"pm25": [8.0, 8.1, 8.2]},  # keep="last" -> 11:00 får 8.2
        index=[tid[0], tid[1], tid[1]],  # 11:00 duplicerad
    )
    vader = pd.DataFrame({"Lufttemperatur": [5.0, 5.1]}, index=tid)

    T_obs = bestam_ankarpunkt(pm25, vader)

    assert T_obs == pd.Timestamp("2026-10-21 11:00", tz="Europe/Stockholm")


def _uppf_df(n, start="2026-10-21 10:00"):
    """Rättad uppföljnings-DataFrame, samma kolumner som folj_uppa bygger."""
    tid = pd.date_range(start, periods=n, freq="h", tz="Europe/Stockholm")
    return pd.DataFrame(
        {
            "körningstid": tid,
            "linje": ["vanlig" if i % 2 == 0 else "forskjuten" for i in range(n)],
            "mål_tid": tid,
            "prediktion": [10.0 + i for i in range(n)],
            "faktiskt": [9.5 + i for i in range(n)],
        }
    )


def test_visa_senaste_prediktioner_visar_senaste_i_kronologisk_ordning(capsys):
    """Tabellen visar de n senaste raderna, äldst överst."""
    visa_senaste_prediktioner(_uppf_df(15), n=10)

    ut = capsys.readouterr().out
    # De 10 senaste raderna har prediktion 15–24; rad 1–5 ska vara borta
    # ("10.0" är första radens prediktion och förekommer inte i visningen).
    assert "15.0" in ut and "24.0" in ut
    assert "10.0" not in ut
    # Kronologisk ordning: 15:00 (äldsta visas raden) före 00:00 (senaste).
    assert ut.index("2026-10-21 15:00") < ut.index("2026-10-22 00:00")


def test_visa_senaste_prediktioner_faerre_rader_visar_alla(capsys):
    """Färre än n realiserade rader -> alla visas."""
    visa_senaste_prediktioner(_uppf_df(4), n=10)

    ut = capsys.readouterr().out
    assert "13.0" in ut and "12.5" in ut
    assert "10.0" in ut  # första raden finns kvar när alla rader visas


# ----------------------------------------------------------
# Nisch-routern (plan 01)
# ----------------------------------------------------------

class _NischDummy:
    """Picklbar dummy-modell för nisch-champion-filer i testerna."""
    def predict(self, X):
        return X["pm25"].to_numpy() * 3


def test_bestam_vanlig_champion_routar_vid_och_over_gransen():
    """Gränsen är inklusiv (>=) — precis som nisch-träningens radfilter:
    ett värde under gränsen ger championen, på/över ger nisch-championen."""
    vanlig_modell, vanlig_namn = object(), "SVR vanlig (T+6)"
    nisch_modell, nisch_namn = object(), "SVR nisch (T+6)"
    grans = 14.4

    # 1 µg/m³ under gränsen -> ordinarie champion
    modell, namn, anvand = bestam_vanlig_champion(
        vanlig_modell, vanlig_namn, nisch_modell, nisch_namn, grans,
        pm25_varde=grans - 1.0,
    )
    assert (modell, namn, anvand) == (vanlig_modell, vanlig_namn, None)

    # Exakt på gränsen -> nisch-champion (OBS: object()-identiteten bara
    # för att stärka att det är rätt objekt som returneras)
    modell, namn, anvand = bestam_vanlig_champion(
        vanlig_modell, vanlig_namn, nisch_modell, nisch_namn, grans,
        pm25_varde=grans,
    )
    assert (modell, namn, anvand) == (nisch_modell, nisch_namn, grans)

    # Över gränsen -> nisch-champion
    modell, namn, anvand = bestam_vanlig_champion(
        vanlig_modell, vanlig_namn, nisch_modell, nisch_namn, grans,
        pm25_varde=25.0,
    )
    assert (modell, namn, anvand) == (nisch_modell, nisch_namn, grans)


def test_bestam_vanlig_champion_utan_nisch_champion():
    """Ingen nisch-champion (modell=None) -> routern är av och den
    ordinarie championen servar alltid, även vid mycket höga halter."""
    vanlig_modell, vanlig_namn = object(), "SVR vanlig (T+6)"

    modell, namn, anvand = bestam_vanlig_champion(
        vanlig_modell, vanlig_namn, None, None, None, pm25_varde=500.0,
    )

    assert (modell, namn, anvand) == (vanlig_modell, vanlig_namn, None)


def _bygg_nisch_champion(tmp_path, threshold=14.4, med_grans=True):
    """Skapar modellfil + metadata + champion-logg för nisch-linjen i
    tmp_path och returnerar en cfg-namespace (SimpleNamespace)."""
    sokvag = save_champion_model(
        model=_NischDummy(),
        model_name="Nisch Dummy nisch (T+6)",
        best_row=pd.Series({"Modell": "Nisch Dummy nisch (T+6)", "Validation High PM2.5 MAE": 1.0}),
        models_dir=tmp_path,
        random_state=42,
        metric="Validation High PM2.5 MAE",
        filprefix="nisch_",
        threshold=threshold if med_grans else None,
        threshold_rule="pm25_p90_training" if med_grans else None,
    )
    append_champion_log(
        log_path=tmp_path / "champion_nisch.csv",
        model_name="Nisch Dummy nisch (T+6)",
        metric="Validation High PM2.5 MAE",
        score=1.0,
        sokvag=sokvag,
    )
    return SimpleNamespace(
        nisch_log=tmp_path / "champion_nisch.csv",
        models_dir=tmp_path,
    )


def test_hamta_nisch_champion_laeser_grans_ur_metadata(tmp_path):
    """Gränsen läses ur championens metadata — inte ur config/minnet."""
    cfg = _bygg_nisch_champion(tmp_path, threshold=14.4)

    modell, namn, grans = hamta_nisch_champion(cfg)

    assert namn == "Nisch Dummy nisch (T+6)"
    assert grans == 14.4
    # Modellen laddas från rätt fil (dummy predikterar pm25 * 3)
    X = pd.DataFrame({"pm25": [10.0]})
    assert modell.predict(X)[0] == pytest.approx(30.0)


def test_hamta_nisch_champion_utan_grans_stanger_av_routern(tmp_path):
    """Saknas gränsen i metadata (t.ex. gammal fil) fail-säkrar routern
    av — (None, None, None) — istället för att gissa eller krascha."""
    cfg = _bygg_nisch_champion(tmp_path, med_grans=False)

    assert hamta_nisch_champion(cfg) == (None, None, None)
    # Modellen får FÖRSÖKT laddas i testerna? Nej: fil och logg finns,
    # men gränsen saknas -> av. Kontrollera också att loggen finns
    # (fail-safe-testet ska inte slå in på annat sätt).
    assert cfg.nisch_log.is_file()


def test_hamta_nisch_champion_utan_champion(tmp_path):
    """Ingen nisch-logg alls -> (None, None, None)."""
    cfg = SimpleNamespace(
        nisch_log=tmp_path / "finns_inte.csv",
        models_dir=tmp_path,
    )
    assert hamta_nisch_champion(cfg) == (None, None, None)


def _uppf_regim_df():
    """Rättad uppföljning med tre (linje, modell)-grupper: vanlig
    champion-era, nisch-era (routing, plan 01) och forskjuten."""
    tid = pd.date_range("2026-11-01 10:00", periods=6, freq="h",
                        tz="Europe/Stockholm")
    return pd.DataFrame({
        "körningstid": tid,
        "linje": ["vanlig"] * 3 + ["vanlig"] * 1 + ["forskjuten"] * 2,
        "modell": (["SVR vanlig (T+6)"] * 3 + ["SVR nisch (T+6)"]
                   + ["SVR forskjuten (T+6)"] * 2),
        "mål_tid": tid,
        "prediktion": [10.0, 11.0, 12.0, 40.0, 13.0, 14.0],
        "faktiskt": [9.5, 10.5, 11.5, 39.0, 12.5, 13.5],
        "pm25_senaste": [9.5, 10.5, 11.5, 39.0, 12.5, 13.5],
    })


def test_sammanstall_live_uppfoljning_delar_regimer():
    """Nisch-prediktioner får EGNA rader i sammanställningen —
    groupby (linje, modell) delar upp champion- och nisch-era."""
    sammanstallning = sammanstall_live_uppfoljning(_uppf_regim_df())

    grupper = sammanstallning.set_index(
        ["linje", "modell"]
    ).index.tolist()
    assert ("vanlig", "SVR vanlig (T+6)") in grupper
    assert ("vanlig", "SVR nisch (T+6)") in grupper
    assert ("forskjuten", "SVR forskjuten (T+6)") in grupper

    # Rätt antal per grupp + persistence-rad
    antal = sammanstallning.set_index(["linje", "modell"])["antal"].to_dict()
    assert antal[("vanlig", "SVR vanlig (T+6)")] == 3
    assert antal[("vanlig", "SVR nisch (T+6)")] == 1
    assert antal[("forskjuten", "SVR forskjuten (T+6)")] == 2
    assert (
        "persistence-baseline" in sammanstallning["linje"].tolist()
    )


# ----------------------------------------------------------
# Plan 04: skipp-logg, vakter och prognos-kraschens regression
# ----------------------------------------------------------

VADER_KOLUMNER = [
    "Lufttemperatur", "Vindriktning", "Vindhastighet",
    "Relativ luftfuktighet", "Nederbördsmängd",
]

FEATURES = [
    "pm25", "Lufttemperatur", "vind_sin", "vind_cos", "Vindhastighet",
    "Relativ luftfuktighet", "Nederbördsmängd", "timme_sin", "timme_cos",
    "helg", "helg_timme_sin", "helg_timme_cos",
]


class _FastModell:
    """Picklbar dummy-champion som predikterar ett fast värde."""
    def predict(self, X):
        return [42.0] * len(X)


def _vaderdf(index, fukt=70.0):
    """Komplett väderdata (SMHI-format) på ett heltimmesindex."""
    return pd.DataFrame(
        {
            "Lufttemperatur": [5.0] * len(index),
            "Vindriktning": [180.0] * len(index),
            "Vindhastighet": [3.0] * len(index),
            "Relativ luftfuktighet": [fukt] * len(index),
            "Nederbördsmängd": [0.0] * len(index),
        },
        index=index,
    )


def _bygg_champion(models_dir, logg_path, namn, filprefix):
    """Sparar en filbar champion + champion-logg i stil med pipelinen."""
    sokvag = save_champion_model(
        model=_FastModell(),
        model_name=namn,
        best_row=pd.Series({"Modell": namn, "Validation Composite": 1.0}),
        models_dir=models_dir,
        random_state=42,
        metric="Validation Composite",
        filprefix=filprefix,
    )
    append_champion_log(
        log_path=logg_path,
        model_name=namn,
        metric="Validation Composite",
        score=1.0,
        sokvag=sokvag,
    )


def _live_cfg(tmp_path, med_champions=True):
    """cfg-namespace med allt kor_live_prediktion + stada_prognos_dir
    behöver. Hämtningarna (hamta_*) mockas per test."""
    models_dir = tmp_path / "models"
    if med_champions:
        _bygg_champion(
            models_dir, tmp_path / "champion_vanlig.csv",
            "SVR vanlig (T+6)", "",
        )
        _bygg_champion(
            models_dir, tmp_path / "champion_forskjuten.csv",
            "SVR forskjuten (T+6)", "forskjuten_",
        )
    return SimpleNamespace(
        horisont=6,
        features=FEATURES,
        forskjuten_vaderkolumner=VADER_KOLUMNER,
        prognos_dir=tmp_path / "prognoser",
        prediktioner_log=tmp_path / "prognoser" / "prediktioner_live.csv",
        skippad_log=tmp_path / "live_skippad.csv",
        champion_log=tmp_path / "champion_vanlig.csv",
        forskjuten_log=tmp_path / "champion_forskjuten.csv",
        nisch_log=tmp_path / "champion_nisch.csv",  # saknas -> routern av
        models_dir=models_dir,
        max_alder_timmar=6,
        idempotent=True,
        prognos_arkiv_timmar=24,
        prognos_bibehall_dagar=30,
        pm25_arkiv_bibehall_dagar=7,
    )


def _mocka_hamtningar(monkeypatch, nu, prognos_timmar=24, pm25_timmar_bak=48,
                      pm25_nan=False, fukt=70.0):
    """Mockar prognos/pm25/väderobs-hämtningarna runt en fast 'nu'.

    Returnerar den T_obs körningen kommer att välja: senaste hela
    timmen med komplett data (här: nu avrundat ner till hel timme).
    """
    T_obs = nu.floor("h")

    obs_tid = pd.date_range(
        end=T_obs, periods=pm25_timmar_bak, freq="h",
    )
    pm25_df = pd.DataFrame(
        {"pm25": [float("nan")] * len(obs_tid) if pm25_nan else [8.0] * len(obs_tid)},
        index=obs_tid,
    )
    vaderobs_df = _vaderdf(obs_tid, fukt=fukt)

    prognos_tid = pd.date_range(
        start=T_obs, periods=prognos_timmar, freq="h",
    )
    prognos_df = _vaderdf(prognos_tid)
    referenstid = nu.tz_convert("UTC")

    monkeypatch.setattr(pp, "hamta_prognos", lambda cfg: (prognos_df, referenstid))
    monkeypatch.setattr(
        pp, "hamta_senaste_pm25", lambda cfg, timmar_bak=48: pm25_df,
    )
    monkeypatch.setattr(
        pp, "hamta_senaste_vaderobs", lambda cfg, timmar_bak=48: vaderobs_df,
    )
    return T_obs


def test_live_normal_korning_skriver_ingen_skippad(tmp_path, monkeypatch):
    """En normal körning predikterar båda linjerna OCH skriver ingen
    skipp-rad — loggen ska bara innehålla verkliga överhoppningar."""
    cfg = _live_cfg(tmp_path)
    _mocka_hamtningar(monkeypatch, pd.Timestamp.now(tz="Europe/Stockholm"))

    resultat = kor_live_prediktion(cfg)

    assert resultat is not None
    assert list(resultat["linje"]) == ["vanlig", "forskjuten"]
    assert not cfg.skippad_log.is_file()

    logg = pd.read_csv(cfg.prediktioner_log)
    assert list(logg["linje"]) == ["vanlig", "forskjuten"]
    assert (logg["prediktion"] == 42.0).all()


def test_live_prognos_saknas_maltid_servar_vanlig_linje(tmp_path, monkeypatch):
    """Regression (plan 04.3): saknas prognosraden för måltiden kraschar
    INTE längre — den vanliga linjen (behöver bara observationen vid
    T_obs) predicteras och loggas, den förskjutna linjen hoppas över
    med orsak prognos_saknas_maltid."""
    cfg = _live_cfg(tmp_path)
    _mocka_hamtningar(
        monkeypatch, pd.Timestamp.now(tz="Europe/Stockholm"),
        prognos_timmar=3,  # täcker T_obs..T_obs+2 men inte måltiden +6
    )

    # FÅR INTE kasta — tidigare raise:ade bygg_live_rader ValueError
    # och dödade BÅDA linjerna.
    resultat = kor_live_prediktion(cfg)

    assert resultat is not None
    assert list(resultat["linje"]) == ["vanlig"]

    # Den förskjutna linjen: exakt en maskinläsbar skip-rad
    skip = pd.read_csv(cfg.skippad_log)
    assert len(skip) == 1
    assert skip["orsak"].iloc[0] == "prognos_saknas_maltid"
    assert skip["linje"].iloc[0] == "forskjuten"
    assert skip["alinje"].iloc[0] == "linje"

    # ...och den vanliga linjen loggades som vanligt
    logg = pd.read_csv(cfg.prediktioner_log)
    assert list(logg["linje"]) == ["vanlig"]


def test_live_ingen_ankarpunkt_skriver_skiprad(tmp_path, monkeypatch):
    """Vakt 1: ingen komplett timme -> hela körningen hoppas över med
    orsak ingen_ankarpunkt (t_obs hann inte bestämmas -> tomt fält)."""
    cfg = _live_cfg(tmp_path)
    _mocka_hamtningar(
        monkeypatch, pd.Timestamp.now(tz="Europe/Stockholm"), pm25_nan=True,
    )

    resultat = kor_live_prediktion(cfg)

    assert resultat is None
    skip = pd.read_csv(cfg.skippad_log)
    assert skip["orsak"].tolist() == ["ingen_ankarpunkt"]
    assert pd.isna(skip["t_obs"].iloc[0])
    assert not cfg.prediktioner_log.is_file()


def test_live_for_gammal_ankarpunkt_skriver_skiprad(tmp_path, monkeypatch):
    """Vakt 2: T_obs äldre än max_alder_timmar -> skip-rad med orsak
    for_gammal_ankarpunkt OCH den faktiska åldern."""
    nu = pd.Timestamp.now(tz="Europe/Stockholm")
    cfg = _live_cfg(tmp_path)

    # Serien slutar 9 hela timmar tillbaka i tiden: T_obs blir ca 9 h
    # gammal och överstiger max_alder_timmar (6).
    T_obs_gammal = nu.floor("h") - pd.Timedelta(hours=9)
    obs_tid = pd.date_range(end=T_obs_gammal, periods=48, freq="h")
    monkeypatch.setattr(
        pp, "hamta_senaste_pm25",
        lambda cfg, timmar_bak=48: pd.DataFrame(
            {"pm25": [8.0] * len(obs_tid)}, index=obs_tid,
        ),
    )
    monkeypatch.setattr(
        pp, "hamta_senaste_vaderobs",
        lambda cfg, timmar_bak=48: _vaderdf(obs_tid),
    )
    prognos_tid = pd.date_range(start=nu.floor("h"), periods=24, freq="h")
    monkeypatch.setattr(
        pp, "hamta_prognos",
        lambda cfg: (_vaderdf(prognos_tid), nu.tz_convert("UTC")),
    )

    resultat = kor_live_prediktion(cfg)

    assert resultat is None
    skip = pd.read_csv(cfg.skippad_log)
    assert skip["orsak"].tolist() == ["for_gammal_ankarpunkt"]
    assert skip["pm25_alder_h"].iloc[0] >= 9.0
    assert pd.notna(skip["t_obs"].iloc[0])  # t_obs är ifyllt
    assert not cfg.prediktioner_log.is_file()


def test_live_t_obs_redan_predikterad_skriver_skiprad(tmp_path, monkeypatch):
    """Vakt 3: idempotensen. T_obs redan loggad -> skip-rad med orsak
    t_obs_redan_predikterad (t_obs och ålder med i raden)."""
    nu = pd.Timestamp.now(tz="Europe/Stockholm")
    cfg = _live_cfg(tmp_path)
    T_obs = _mocka_hamtningar(monkeypatch, nu)

    # Simulera en tidigare körning för exakt samma T_obs
    cfg.prediktioner_log.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "körningstid": nu.isoformat(),
        "t_obs": T_obs.isoformat(),
        "mål_tid": (T_obs + pd.Timedelta(hours=6)).isoformat(),
        "linje": "vanlig",
        "modell": "SVR vanlig (T+6)",
        "prediktion": 42.0,
        "pm25_senaste": 8.0,
    }]).to_csv(cfg.prediktioner_log, index=False)

    resultat = kor_live_prediktion(cfg)

    assert resultat is None
    skip = pd.read_csv(cfg.skippad_log)
    assert skip["orsak"].tolist() == ["t_obs_redan_predikterad"]
    assert pd.notna(skip["pm25_alder_h"].iloc[0])
    assert pd.notna(skip["t_obs"].iloc[0])


def test_live_champion_saknas_per_linje(tmp_path, monkeypatch):
    """Vakt 4: saknas champion för en linje loggas det per linje med
    orsak champion_saknas:<linje> — och endast den linjen hoppas över."""
    cfg = _live_cfg(tmp_path, med_champions=False)
    _mocka_hamtningar(monkeypatch, pd.Timestamp.now(tz="Europe/Stockholm"))

    # Förskjuten champion finns, vanlig saknas
    _bygg_champion(
        tmp_path / "models", tmp_path / "champion_forskjuten.csv",
        "SVR forskjuten (T+6)", "forskjuten_",
    )

    resultat = kor_live_prediktion(cfg)

    assert resultat is not None
    assert list(resultat["linje"]) == ["forskjuten"]

    skip = pd.read_csv(cfg.skippad_log)
    assert skip["orsak"].tolist() == ["champion_saknas:vanlig"]
    assert skip["linje"].iloc[0] == "vanlig"


def test_live_ogiltig_vaderslot_degraderar_linjen(tmp_path, monkeypatch):
    """sanity_kolla_vader gör en slot ofullständig (fukt > 100) -> den
    linjen hoppas över (orsak vader_slot_ogiltig:<linje>) i stället för
    att ett NaN flödar rakt in i modellen; den andra linjen servas."""
    cfg = _live_cfg(tmp_path)
    _mocka_hamtningar(
        monkeypatch,
        pd.Timestamp.now(tz="Europe/Stockholm"),
        fukt=150.0,  # OGILTIG relativ luftfuktighet (> 100)
    )

    resultat = kor_live_prediktion(cfg)

    # Observationen (vanlig) är ogiltig — prognosen (forskjuten) är hel
    assert resultat is not None
    assert list(resultat["linje"]) == ["forskjuten"]

    skip = pd.read_csv(cfg.skippad_log)
    assert skip["orsak"].tolist() == ["vader_slot_ogiltig:vanlig"]
    assert skip["linje"].iloc[0] == "vanlig"

    logg = pd.read_csv(cfg.prediktioner_log)
    assert list(logg["linje"]) == ["forskjuten"]


# ----------------------------------------------------------
# Plan 04: skip-sammanfattningen i uppföljningen
# ----------------------------------------------------------

def _skipplog_rad(**over):
    rad = {
        "timestamp": pd.Timestamp.now().isoformat(timespec="seconds"),
        "t_obs": "", "orsak": "for_gammal_ankarpunkt",
        "linje": "", "alinje": "hela körningen", "pm25_alder_h": "",
    }
    rad.update(over)
    return rad


def test_visa_skip_sammanstallning_raknar_per_orsak(tmp_path, capsys):
    """Sammanfattningen räknar per orsak inom fönstret och ignorerar
    rader äldre än fönstret — "varför är det så få prediktioner?" på
    en blick."""
    logg = tmp_path / "live_skippad_6247.csv"
    pd.DataFrame([
        _skipplog_rad(),
        _skipplog_rad(),
        # Utanför 7-dagarsfönstret: räknas inte
        _skipplog_rad(
            timestamp=(pd.Timestamp.now() - pd.Timedelta(days=9))
            .isoformat(timespec="seconds"),
            orsak="t_obs_redan_predikterad",
        ),
    ]).to_csv(logg, index=False)

    summ = visa_skip_sammanstallning(
        SimpleNamespace(skippad_log=logg), timmar_bak=24 * 7,
    )

    assert summ is not None
    antal = dict(zip(summ["orsak"], summ["antal"]))
    assert antal == {"for_gammal_ankarpunkt": 2}

    ut = capsys.readouterr().out
    assert "Overhoppade" in ut or "Överhoppade" in ut
    assert "for_gammal_ankarpunkt" in ut
    assert "t_obs_redan_predikterad" not in ut


def test_visa_skip_sammanstallning_saknad_logg(tmp_path, capsys):
    """Ingen skipp-logg = 0 loggade överhopp — ett uttryckligt svar,
    inte tystnad."""
    summ = visa_skip_sammanstallning(
        SimpleNamespace(skippad_log=tmp_path / "finns_inte.csv"),
        timmar_bak=24 * 7,
    )

    assert summ is None
    assert "0 loggade överhopp" in capsys.readouterr().out


def test_visa_skip_sammanstallning_tomma_fonstret(tmp_path, capsys):
    """Loggen finns men inget ligger i fönstret -> explicit nollrad."""
    logg = tmp_path / "live_skippad_6247.csv"
    pd.DataFrame([
        _skipplog_rad(
            timestamp=(pd.Timestamp.now() - pd.Timedelta(days=30))
            .isoformat(timespec="seconds"),
        ),
    ]).to_csv(logg, index=False)

    summ = visa_skip_sammanstallning(
        SimpleNamespace(skippad_log=logg), timmar_bak=24 * 7,
    )

    assert summ is None
    assert "0." in capsys.readouterr().out or ": 0" in capsys.readouterr().out
