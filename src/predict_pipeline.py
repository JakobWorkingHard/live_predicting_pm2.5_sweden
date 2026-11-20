import pandas as pd

from src.champion import load_latest_champion
from src.extracting_data import (
    hamta_prognos,
    hamta_senaste_pm25,
    hamta_senaste_vaderobs,
    stada_prognos_dir,
)
from src.features import add_wind_features, add_time_features
from pathlib import Path
from src.io import (
    append_prediktioner_log,
    append_skippad_log,
    hamta_champion_modell,
    las_champion_metadata,
)
from src.logger import setup_logger
from src.metrics import display_df, sammanstall_live_uppfoljning
from src.validate_process import sanity_kolla_vader


log = setup_logger()

def bestam_ankarpunkt(pm25_df: pd.DataFrame, vaderobs_df: pd.DataFrame):
    """
    T_obs = senaste hela timme med komplett pm25 + alla väderparametrar.

    Parningen sker via indexet (heltimmes-tidsstämplar) — samma exakta
    parningsregel som huvudpipelinens merge på "tid", så live och träning
    kan aldrig para ihop data olika. Returnerar None om ingen komplett
    timme finns; kor_live_prediktion (steg 9) hoppar då över körningen.
    """
    # Defensivt: live passerar aldrig validate_data:s dublettkoll, och
    # duplikatetiketter måste bort FÖRE concat — annars kan själva
    # sammanfogningen krascha (reindex på dubbelt index).
    pm25 = pm25_df[~pm25_df.index.duplicated(keep="last")]
    vaderobs = vaderobs_df[~vaderobs_df.index.duplicated(keep="last")]

    gemensam = pd.concat([pm25["pm25"], vaderobs], axis=1).sort_index()

    kompletta = gemensam.dropna()

    if kompletta.empty:
        log.warning("Ingen komplett timme hittades — ingen ankarpunkt.")
        return None

    T_obs = kompletta.index.max()
    log.info("Ankarpunkt T_obs: %s", T_obs)
    return T_obs



def bygg_live_rader(cfg, T_obs, pm25_varde, vaderobs_rad, vader_prognos_df, pm25_alder_h=None):
    """
    Bygger en modellrad per linje (vanlig + forskjuten) för live-prediktion.

    Båda raderna gäller tidpunkten T_obs med samma pm25-mätning; det som
    skiljer är väderslotten — samma slot-alignment som träningen
    (jämför ersatt_vader_med_maltidsvader):
      vanlig:    väderOBSERVATION vid T_obs
      forskjuten: väderPROGNOS för måltiden T_obs + horisont

    Båda slots går genom sanity_kolla_vader (samma intervallregler som
    replace_invalid_values tvingar på träningsdatan), sedan add_wind_
    features + add_time_features. Kolumnerna blir exakt cfg.features,
    i samma ordning som modellerna tränades med.

    Plan 04: saknad data kraschar ALDRIG flödet. Saknas prognosraden
    för måltiden byggs den vanliga linjen ändå (den behöver bara
    observationen vid T_obs) och den förskjutna linjen hoppas över
    med en skip-rad (orsak=prognos_saknas_maltid) i
    [live].skippad_log — tidigare raises ValueError och dödade BÅDA
    linjerna. Samma filosofi för en slot som sanity_kolla_vader gjort
    ofullständig (orsak=vader_slot_ogiltig:<linje>): serva de linjer
    vars slot är komplett, logga de andra — inget NaN når modellerna.

    Returnerar en dict som kan SAKNA nycklar: inte alla linjer kunde
    byggas. Anroparen (steg 5–8) predikterar bara de linjer som finns.
    """
    maltid = T_obs + pd.Timedelta(hours=cfg.horisont)

    # Slot per linje: vanliga linjen behöver bara observationen vid
    # T_obs, den förskjutna bara prognosraden vid måltiden.
    slots = {"vanlig": sanity_kolla_vader(vaderobs_rad)}

    if maltid in vader_prognos_df.index:
        slots["forskjuten"] = sanity_kolla_vader(vader_prognos_df.loc[maltid])
    else:
        log.warning(
            "Prognosen täcker inte måltiden %s (senaste prognossteg: %s) "
            "— den förskjutna linjen hoppas över, den vanliga servas.",
            maltid, vader_prognos_df.index.max(),
        )
        append_skippad_log(
            cfg.skippad_log, orsak="prognos_saknas_maltid", t_obs=T_obs,
            linje="forskjuten", alinje="linje", pm25_alder_h=pm25_alder_h,
        )

    rader = {}
    for namn, slot in slots.items():
        # Plan 04: en slot med ogiltiga/saknade värden (sanity_kolla_vader
        # sätter ogiltiga -> NaN) får INTE matas in i modellen — linjen
        # hoppas över medan de kompletta linjerna servas.
        saknade_varden = [
            kolumn for kolumn in cfg.forskjuten_vaderkolumner
            if pd.isna(slot[kolumn])
        ]
        if saknade_varden:
            log.warning(
                "Väderslotten för linjen %s är inte komplett "
                "(saknas: %s) — linjen hoppas över.",
                namn, ", ".join(saknade_varden),
            )
            append_skippad_log(
                cfg.skippad_log, orsak=f"vader_slot_ogiltig:{namn}",
                t_obs=T_obs, linje=namn, alinje="linje",
                pm25_alder_h=pm25_alder_h,
            )
            continue

        if namn == "forskjuten":
            # Logga väderprognosen som den förskjutna linjen faktiskt
            # matas med (efter sanity_kolla_vader, dvs exakt de värden
            # modellen ser — och kompletthetskontrollen ovan garanterar
            # att inget värde saknas).
            log.info(
                "Väderprognos för måltiden %s (använd av linjen forskjuten): %s",
                maltid,
                ", ".join(
                    f"{kolumn}={slot[kolumn]:g}"
                    for kolumn in cfg.forskjuten_vaderkolumner
                ),
            )

        rad = {"tid": T_obs, "pm25": float(pm25_varde)}
        for kolumn in cfg.forskjuten_vaderkolumner:
            rad[kolumn] = slot[kolumn]

        df_rad = pd.DataFrame([rad])
        df_rad = add_wind_features(df_rad)
        df_rad = add_time_features(df_rad, cfg.horisont)
        rader[namn] = df_rad[list(cfg.features)]

    log.info(
        "Modellrader byggda för T_obs %s, måltid %s: %s.",
        T_obs, maltid, ", ".join(rader),
    )
    return rader


def visa_prognosrad(cfg, prognos_df, maltid):
    """
    Visar väderprognosraden vid måltiden — samma slot som bygg_live_rader
    plockar, sanity-kollad så att tabellen visar exakt de värden modellen
    matas med. Används både i normalflödet (steg 7) och i idempotensvakten
    (tidigare prediktioner för samma T_obs). Bara den förskjutna linjen
    använder prognos; den vanliga linjen predicteras på väderobservationen
    vid T_obs.
    """
    if maltid not in prognos_df.index:
        log.warning("Prognosen täcker inte %s — ingen väderprognos visas.", maltid)
        return

    prognos_rad = sanity_kolla_vader(prognos_df.loc[maltid])[
        list(cfg.forskjuten_vaderkolumner)
    ]
    prognos_visning = prognos_rad.to_frame().T
    prognos_visning.insert(0, "tid (måltid)", f"{maltid:%Y-%m-%d %H:%M %Z}")
    print(
        f"\nVäderprognos för den förskjutna prediktionen "
        f"(SMHI punktprognos, prognostid {maltid:%Y-%m-%d %H:%M %Z}):"
    )
    display_df(prognos_visning)



def hamta_nisch_champion(cfg):
    """
    Läser in nisch-championen (höga PM2.5, plan 01) och dess gräns.

    Gränsen läses ENDAST ur championens metadata (.json) — aldrig ur
    config eller minnet — så att träningen som beräknade gränsen och
    routern som använder den aldrig kan skilja sig åt.

    Returnerar
    -------
    tuple
        (modell, namn, gräns), eller (None, None, None) om ingen
        nisch-champion finns eller metadata saknar gräns (routern av).
    """
    modell, namn, filnamn = load_latest_champion(cfg.nisch_log, cfg.models_dir)
    if modell is None:
        return None, None, None

    grans = las_champion_metadata(cfg.models_dir, filnamn).get("threshold")
    if grans is None:
        log.warning(
            "Nisch-championen %s saknar 'threshold' i metadata — "
            "routern är av och den ordinarie championen servar allt.", namn,
        )
        return None, None, None

    return modell, namn, grans


def bestam_vanlig_champion(
    vanlig_modell, vanlig_namn, nisch_modell, nisch_namn, nisch_grans, pm25_varde,
):
    """
    Routern (plan 01): avgör vilken champion som predikterar den VANLIGA
    linjen den här körningen.

      pm25 vid T_obs >= nisch_grans -> nisch-championen predikterar
      annars                         -> den ordinarie championen

    nisch-modell/namn/gräns får komma sanerade från hamta_nisch_champion:
    är nisch_modell None finns ingen aktiv rutningsregim och den ordinarie
    championen servar alltid. Gränsen är inklusiv (>=), exakt som
    nisch-träningens radfilter — ingen avståndszon mellan träning och live.

    Returnerar
    -------
    tuple
        (modell, namn, anvand_grans), där anvand_grans är gränsen som
        faktiskt avgjorde (None när ingen routing skedde) — loggas som
        nisch_threshold i prediktionsloggen.
    """
    if nisch_modell is None:
        return vanlig_modell, vanlig_namn, None

    if pm25_varde >= nisch_grans:
        log.info(
            "Routing: pm25 %.1f µg/m³ >= gränsen %.2f µg/m³ — "
            "nisch-championen %s predikterar den vanliga linjen.",
            pm25_varde, nisch_grans, nisch_namn,
        )
        return nisch_modell, nisch_namn, nisch_grans

    return vanlig_modell, vanlig_namn, None


def kor_live_prediktion(cfg):
    """
    Kör live-flödet en gång: hämta → ankarpunkt → vakter → modellrader →
    ladda championer → prediktera → visa → logga → städa arkivet.

    Arkivstädningen (stada_prognos_dir) körs i en finally-sats, så att
    den gamla prognos- och pm25-arkivering som samlas vid varje körning
    hålls begränsad också när själva prediktionen hoppas över av någon
    vakt (tidiga return-satser i _kor_live_prediktion).

    Returnerar DataFrame med en rad per FAKTISKT predikterad linje, eller
    None om alla vakter/linjer hoppades över — sedan plan 04 skrivs varje
    överhoppning som en maskinläsbar rad i [live].skippad_log (inte bara
    en loggvarning), så fylls uppföljningens stickprov inte tyst.
    """
    try:
        return _kor_live_prediktion(cfg)
    finally:
        stada_prognos_dir(cfg)


def _kor_live_prediktion(cfg):
    """
    Själva live-flödet. Vakter (i ordning):
      1. Ingen komplett timme (bestam_ankarpunkt -> None) → hoppa över
      2. T_obs äldre än cfg.max_alder_timmar → hoppa över
      3. Idempotens: T_obs redan loggad (om cfg.idempotent) → hoppa över
      4. Per linje: champion saknas → logga varning, hoppa över linjen

    Plan 01: finns en nisch-champion avgör routern
    (bestam_vanlig_champion) vilken champion som predikterar den
    vanliga linjen — pm25 vid T_obs över gränsen ger nisch-championen.

    Plan 04: varje vakt/linje-överhoppning skriver exakt en
    orsaksrad i [live].skippad_log (orsak enligt plan 04:s enum), och
    bygg_live_rader degraderar per linje vid saknad prognos/väderslot
    i stället för att krascha hela körningen.
    """
    nu = pd.Timestamp.now(tz="Europe/Stockholm")

    # 1. Hämta prognos + senaste observationer
    # Notera att referenstiden är NÄR prognosen skapades hos SMHI, dvs en timestamp för hur färsk prognosen är
    prognos_df, referenstid = hamta_prognos(cfg)
    pm25_df = hamta_senaste_pm25(cfg)
    vaderobs_df = hamta_senaste_vaderobs(cfg)

    # 2. Ankartid
    T_obs = bestam_ankarpunkt(pm25_df, vaderobs_df)
    if T_obs is None:
        log.warning("Live-körning hoppades över: ingen komplett timme.")
        # Plan 04: varje överhoppning är maskinläsbar, inte bara en
        # loggvarning — t_obs är ännu inte bestämt och lämnas tomt.
        append_skippad_log(cfg.skippad_log, orsak="ingen_ankarpunkt")
        return None

    alder_h = (nu - T_obs).total_seconds() / 3600
    maltid = T_obs + pd.Timedelta(hours=cfg.horisont)
    log.info("T_obs: %s (%.1f h gammal) -> prediktion för %s", T_obs, alder_h, maltid)

    # 3. Vakter
    if alder_h > cfg.max_alder_timmar:
        log.warning(
            "Live-körning hoppades över: T_obs %.1f h gammal (max %s h).",
            alder_h, cfg.max_alder_timmar,
        )
        append_skippad_log(
            cfg.skippad_log, orsak="for_gammal_ankarpunkt",
            t_obs=T_obs, pm25_alder_h=alder_h,
        )
        return None

    if cfg.idempotent and cfg.prediktioner_log.is_file():
        tidigare = pd.read_csv(cfg.prediktioner_log)
        gamla_t_obs = pd.to_datetime(tidigare["t_obs"], utc=True)
        if (gamla_t_obs == T_obs).any():
            gamla = tidigare[gamla_t_obs == T_obs]
            log.warning(
                "Live-körning hoppades över: T_obs redan predikterad "
                "(slå av [live].idempotent för att köra ändå)."
            )
            log.info("Tidigare prediktioner för T_obs %s:", T_obs)
            display_df(
                gamla[["linje", "modell", "prediktion", "mål_tid", "pm25_senaste"]]
            )
            # Visa väderprognosen för samma måltid under de tidigare
            # raderna — OBS: denna körnings prognos, alltså inte
            # nödvändigtvis samma prognos som den ursprungliga körningen
            # använde (prognos_referenstid i loggen avgör det).
            visa_prognosrad(cfg, prognos_df, maltid)
            append_skippad_log(
                cfg.skippad_log, orsak="t_obs_redan_predikterad",
                t_obs=T_obs, pm25_alder_h=alder_h,
            )
            return None

    # 4. Modellrader för båda linjerna (rieal kedja: steg 5–8).
    # Plan 04: bygg_live_rader kan returnera FÄRRE nycklar — saknas
    # prognosraden för måltiden eller en komplett väderslot hoppas den
    # linjen över med en egen skip-rad (ingen ValueError, ingen krasch).
    rader = bygg_live_rader(
        cfg=cfg,
        T_obs=T_obs,
        pm25_varde=pm25_df.loc[T_obs, "pm25"],
        vaderobs_rad=vaderobs_df.loc[T_obs],
        vader_prognos_df=prognos_df,
        pm25_alder_h=alder_h,
    )

    # 5. Ladda championer (olika tupellängd! )
    vanlig_modell, vanlig_namn = hamta_champion_modell(cfg.models_dir, cfg.champion_log)
    forskjuten_modell, forskjuten_namn, _ = load_latest_champion(
        cfg.forskjuten_log, cfg.models_dir
    )

    # 5b. Routern (plan 01): finns en nisch-champion och pm25 vid T_obs
    # är ovanligt högt (>= gränsen i championens metadata) predikterar
    # den VANLIGA linjen med nisch-championen. Forskjuten-linjen
    # routas medvetet inte ännu (egen nisch-modell som uppföljning,
    # se plan 01).
    nisch_modell, nisch_namn, nisch_grans = hamta_nisch_champion(cfg)
    vanlig_modell, vanlig_namn, anvand_nisch_grans = bestam_vanlig_champion(
        vanlig_modell=vanlig_modell,
        vanlig_namn=vanlig_namn,
        nisch_modell=nisch_modell,
        nisch_namn=nisch_namn,
        nisch_grans=nisch_grans,
        pm25_varde=float(pm25_df.loc[T_obs, "pm25"]),
    )

    linjer = [
        ("vanlig", vanlig_modell, vanlig_namn),
        ("forskjuten", forskjuten_modell, forskjuten_namn),
    ]


    poster = []
    for linje, modell, namn in linjer:
        if modell is None:
            log.warning("Hoppar över linjen %s: ingen champion hittades.", linje)
            append_skippad_log(
                cfg.skippad_log, orsak=f"champion_saknas:{linje}",
                t_obs=T_obs, linje=linje, alinje="linje",
                pm25_alder_h=alder_h,
            )
            continue

        if linje not in rader:
            # Plan 04: slot saknades redan i bygg_live_rader (prognos
            # saknas för måltiden eller ogiltig väderslot) — skip-raden
            # är redan loggad DÄR; hoppa över utan att logga en andra
            # gång (exakt en rad per hoppat försök).
            continue

        poster.append({
            "körningstid": nu.isoformat(),
            "t_obs": T_obs.isoformat(),
            "mål_tid": maltid.isoformat(),
            "pm25_senaste": float(pm25_df.loc[T_obs, "pm25"]),
            "pm25_alder_h": round(alder_h, 2),
            "horisont": cfg.horisont,
            "prognos_referenstid": referenstid.isoformat(),
            "linje": linje,
            "modell": namn,
            "prediktion": float(modell.predict(rader[linje])[0]),
            # Gränsen routern använde (plan 01); tom när linjen inte
            # routades. Namnet i 'modell' bär redan "nisch", så
            # uppföljningen delar upp regimerna per (linje, modell).
            "nisch_threshold": (
                anvand_nisch_grans if linje == "vanlig" else float("nan")
            ),
        })

    if not poster:
        log.warning("Inga prediktioner gjordes — inget loggas.")
        return None

    # 6. Visa och logga en rad per linje
    resultat = pd.DataFrame(poster)
    display_df(resultat[["linje", "modell", "prediktion", "mål_tid", "pm25_senaste"]])

    # 7. Visa väderprognosen som den förskjutna linjen predikterades med.
    visa_prognosrad(cfg, prognos_df, maltid)

    append_prediktioner_log(cfg.prediktioner_log, resultat)

    log.info("Loggade %d live-prediktioner i %s.", len(resultat), cfg.prediktioner_log)
    return resultat



def visa_senaste_prediktioner(uppf: pd.DataFrame, n: int = 10):
    """
    Visar de n senaste REALISERADE prediktionerna rad för rad: prediktion
    mot faktiskt mätvärde plus fel, i kronologisk ordning (äldst överst).

    Sortering sker på mål_tid med körningstid som tiebreaker, så raderna
    motsvarar exakt det som skedde senast i loggen. Finns färre än n
    rader visas alla.
    """
    senaste = uppf.sort_values(["mål_tid", "körningstid"]).tail(n).copy()

    senaste["mål_tid"] = (
        senaste["mål_tid"]
        .dt.tz_convert("Europe/Stockholm")
        .dt.strftime("%Y-%m-%d %H:%M")
    )
    senaste["fel"] = (senaste["prediktion"] - senaste["faktiskt"]).round(2)
    for kolumn in ["prediktion", "faktiskt", "fel"]:
        senaste[kolumn] = senaste[kolumn].map(lambda varde: f"{varde:.2f}")

    print(f"\nDe {len(senaste)} senaste realiserade prediktionerna (prediktion vs faktiskt):")
    display_df(
        senaste[["mål_tid", "linje", "prediktion", "faktiskt", "fel"]],
        disable_numparse=True,
        colalign=("left", "left", "right", "right", "right"),
    )


def visa_skip_sammanstallning(cfg, timmar_bak: int = 24 * 7):
    """
    Plan 04: visar antalet överhoppade live-körningar per orsak inom
    det senaste `timmar_bak`-fönstret, läs ur parets egen skipp-logg.

    Syftet är att "varför är det så få prediktioner?" ska gå att besvara
    på en blick — prestationstabellen räknar bara det som faktiskt
    GJORTS, överhoppningarna gör stickprovet ärligt. Därför visas
    sammanfattningen även när ingen prediktion hunnit realiseras.

    Returnerar
    -------
    pd.DataFrame eller None
        Sammanställningen (orsak, antal), eller None om loggen saknas,
        är tom eller inga rader ligger inom fönstret.
    """
    skipp_log = getattr(cfg, "skippad_log", None)
    if skipp_log is None or not Path(skipp_log).is_file():
        print("\nÖverhoppade live-körningar: ingen skipp-logg (0 loggade överhopp).")
        return None

    logg = pd.read_csv(skipp_log)
    if logg.empty or "orsak" not in logg.columns:
        print("\nÖverhoppade live-körningar: 0 loggade överhopp.")
        return None

    # timestamp skrivs som lokal naiv ISO (append_skippad_log) — jämför
    # mot lokal naiv "nu", inte tz-medveten, annars kraschar jämförelsen.
    logg["timestamp"] = pd.to_datetime(logg["timestamp"])
    fonster_start = pd.Timestamp.now() - pd.Timedelta(hours=timmar_bak)
    nyligen = logg[logg["timestamp"] >= fonster_start]

    if nyligen.empty:
        print(
            f"\nÖverhoppade live-körningar senaste {timmar_bak // 24}"
            " dagarna: 0."
        )
        return None

    sammanstallning = (
        nyligen["orsak"]
        .value_counts()
        .rename("antal")
        .rename_axis("orsak")
        .reset_index()
    )
    print(
        "\nÖverhoppade live-körningar senaste "
        f"{timmar_bak // 24} dagarna (orsak, antal):"
    )
    display_df(sammanstallning)
    return sammanstallning


def folj_uppa(cfg, timmar_bak: int = 24 * 7, senaste: int = 10):
    """
    Rättar loggade live-prediktioner mot verkliga PM2.5-mätningar och
    sammanställer MAE/RMSE/R²/Bias per linje + persistence-baseline.

    Visar först de `senaste` senaste realiserade prediktionerna rad för
    rad (visa_senaste_prediktioner), därefter den aggregerade
    sammanställningen.

    Plan 04: överhoppade körningar per orsak (visa_skip_sammanstallning)
    visas i samma utskrift — även när inget hunnit realiseras, för det
    är precis då frågan "varför syns inget?" är som viktigast.

    Returnerar sammanställningen (DataFrame), eller None om loggen saknas
    eller ingen prediktion hunnit realiseras ännu.
    """
    # Plan 04: skipp-sammanfattningen först — den trycks oavsett vilken
    # väg uppföljningen tar nedan (tysta tidiga return-satser ska inte
    # också tysta bort överhoppningarna).
    visa_skip_sammanstallning(cfg, timmar_bak=timmar_bak)

    if not cfg.prediktioner_log.is_file():
        log.warning("Ingen prediktionslogg att följa upp: %s", cfg.prediktioner_log)
        return None

    logg = pd.read_csv(cfg.prediktioner_log)
    for kolumn in ["körningstid", "t_obs", "mål_tid"]:
        logg[kolumn] = pd.to_datetime(logg[kolumn], utc=True)

    # Verkliga värden, indexerade i UTC för enkel matchning mot mål_tid
    verkligt = hamta_senaste_pm25(cfg, timmar_bak=timmar_bak)["pm25"]
    verkligt.index = verkligt.index.tz_convert("UTC")
    logg["faktiskt"] = logg["mål_tid"].dt.tz_convert("UTC").map(verkligt)

    uppf = logg.dropna(subset=["faktiskt"]).copy()

    if uppf.empty:
        log.info("Ingen prediktion har hunnit realiseras ännu.")
        return None

    visa_senaste_prediktioner(uppf, n=senaste)

    sammanstallning = sammanstall_live_uppfoljning(uppf)
    log.info("Uppföljning: %d rättade prediktioner.", len(uppf))
    display_df(sammanstallning)
    return sammanstallning