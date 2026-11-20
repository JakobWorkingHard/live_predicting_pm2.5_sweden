"""
Stationparsmodulen: upptäcker alla PM2.5-stationer och väderstationer,
parar ihop dem (närmaste väderstation inom max_km) och cachar resultatet.

Körs ENDAST inifrån main()/live() — modulen gör ingenting vid import.

Regler (beslut 2026-11-15):
  * PM2.5-stationen är "huvudstationen": varje PM2.5-station paras med
    sin NÄRMASTE kvalificerade väderstation inom [stations].max_km.
  * En väderstation FÅR serva flera PM2.5-stationer (närmaste-välaren
    arbetar per PM2.5-station, utan exklusivitet).
  * PM2.5-serien måste aktivt rapportera: senaste värdet (lastValue)
    yngre än [stations].max_serie_alder_dagar, annars hoppas stationen över.
  * Minst [stations].min_traningsmanader hela månader träningsdata
    måste finnas EFTER att valideringsperiodens månader reserverats.
  * Väderstationen måste mäta ALLA fem parametrar och ha data som
    täcker hela det planerade träningsfönstret (från
    max(seriens start, idag - [time].train_rullande_ar)).

Upptäckten cachas i [stations].stationpar_cache (CSV) och förnyas som
senast var [stations].cache_giltig_dagar dagar gammal — luft-API:t är
långsamt och opålitligt (503:or, ~12 s/anrop noterade 2026-11-15), så
upptäckten ska sällan behöva köras. Vid nätverksfel faller koden
tillbaka på en gammal cache om en sådan finns.
"""

import math
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

from src.logger import setup_logger

log = setup_logger(__name__)

# Luft-API:ets PM2.5-fenomen (52North id) och hur seriens EoI-kod ser ut
# i tidsserie-etiketten, t.ex. "PM2.5 SPP-SE0058A_06001_100_101, ...".
PM25_FENOMEN_ID = "6001"
PM25_ETIKETT_FILTREGEX = ("_06001_",)  # kompletteras med startswith("PM2.5")

TIDZON = "Europe/Stockholm"

CACHE_KOLUMNER = [
    "pm25_station_id", "pm25_namn", "pm25_klass", "pm25_lat", "pm25_lon",
    "tidsserie_id", "serie_start", "serie_slut",
    "vaderstation_id", "vaderstation_namn", "vaderstation_lat",
    "vaderstation_lon", "avstand_km", "beraknad_tid",
]


@dataclass(frozen=True)
class Pm25Station:
    """En PM2.5-station med sin valda (aktivaste) tidsserie."""
    station_id: int
    namn: str
    klass: str
    lat: float
    lon: float
    tidsserie_id: int
    serie_start: str    # "%Y-%m-%d" i svensk tid
    serie_slut: str     # "%Y-%m-%d" i svensk tid


@dataclass(frozen=True)
class Vaderstation:
    """En SMHI-station som mäter alla fem väderparametrar."""
    station_id: int
    namn: str
    lat: float
    lon: float
    data_fran: pd.Timestamp  # första datum ALLA parametrar finns (senaste av dem)
    aktiv: bool


@dataclass(frozen=True)
class StationPar:
    """Ett ihopparat modell-par: PM2.5-station (huvud) + närmaste väderstation."""
    pm25_station_id: int
    pm25_namn: str
    pm25_klass: str
    pm25_lat: float
    pm25_lon: float
    tidsserie_id: int
    serie_start: str
    serie_slut: str
    vaderstation_id: int
    vaderstation_namn: str
    vaderstation_lat: float
    vaderstation_lon: float
    avstand_km: float

    @property
    def etikett(self) -> str:
        return f"{self.pm25_namn} + {self.vaderstation_namn} ({self.avstand_km:.1f} km)"


# ----------------------------------------------------------------
# Hjälpfunktioner
# ----------------------------------------------------------------

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Storcirkelavstånd i kilometer mellan två lat/lon-grader."""
    R = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _api_get(url: str, forsok: int = 4, timeout: tuple = (15, 60)):
    """GET med backoff-omförsök — samma grundmönster som hamta_med_retry."""
    sistafelet = None
    for i in range(forsok):
        try:
            svar = requests.get(url, timeout=timeout)
            if svar.status_code in (502, 503, 504) and i < forsok - 1:
                log.warning("HTTP %s vid %s — försök %d/%d om 10 s.",
                            svar.status_code, url.split("/")[-1], i + 1, forsok)
                time.sleep(10)
                continue
            svar.raise_for_status()
            return svar
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError,
                requests.exceptions.HTTPError) as e:
            sistafelet = e
            if i < forsok - 1:
                time.sleep(5 * (i + 1))
    raise sistafelet


def _datum_lokal(ms: int) -> str:
    """Millisekunder (UTC) -> '%Y-%m-%d' i svensk tid."""
    return pd.to_datetime(ms, unit="ms", utc=True).tz_convert(TIDZON).strftime("%Y-%m-%d")


# ----------------------------------------------------------------
# Upptäckt: PM2.5-stationer
# ----------------------------------------------------------------

def _ar_pm25_tidsserie(etikett: str) -> bool:
    return etikett.startswith("PM2.5") or any(m in etikett for m in PM25_ETIKETT_FILTREGEX)


def _hamta_tidsserie_meta(luft_api: str, tidsserie_id: str) -> dict | None:
    """Metadata (first/last/primärdata) för en tidsserie. None vid miss."""
    try:
        svar = _api_get(f"{luft_api}/timeseries/{tidsserie_id}", forsok=3)
        d = svar.json()
        return {
            "first": (d.get("firstValue") or {}).get("timestamp"),
            "last": (d.get("lastValue") or {}).get("timestamp"),
            "primary": d.get("primaryData"),
            "fenomen": str((d.get("parameters", {}).get("phenomenon") or {}).get("id", "")),
        }
    except Exception as e:
        log.warning("Kunde inte hämta metadata för tidsserie %s: %s", tidsserie_id, e)
        return None


def hamta_pm25_stationer(luft_api: str, max_serie_alder_dagar: int) -> list[Pm25Station]:
    """
    Hämtar alla PM2.5-stationer ur 52North-API:t.

    En station kan ha flera PM2.5-tidsserier (t.ex. Malmö Dalaplans
    gamla serie 126, slut 2024, och nuvarande 6247). Per station väljs
    serien med det SENASTE lastValue-tillfället; stationen hoppas sedan
    över om det senaste värdet är äldre än max_serie_alder_dagar.

    Seriernas first/last-värden hämtas med parallella trådar — detalj-
    endpointen är långsam (~12 s/anrop vid probe 2026-11-15) och serien
    är oberoende. Misslyckade anrop repas i upp till två extrapass.
    """
    svar = _api_get(f"{luft_api}/timeseries", timeout=(15, 240))
    alla = svar.json()
    kandidater = [t for t in alla if _ar_pm25_tidsserie(t["label"])]
    log.info("Luft-API:et: %d tidsserier, varav %d PM2.5-kandidater.",
             len(alla), len(kandidater))

    # station -> {"meta": stationsinfo, "ts": {tidsserie_id: metadata}}
    stationer: dict[int, dict] = {}
    for t in kandidater:
        s = t["station"]
        lon, lat = s["geometry"]["coordinates"][:2]
        post = stationer.setdefault(
            int(s["id"]),
            {"namn": s["properties"]["label"],
             "klass": s["properties"].get("classification", "?"),
             "lat": float(lat), "lon": float(lon), "ts": {}},
        )
        post["ts"][str(t["id"])] = None

    alla_tider = list({tid for post in stationer.values() for tid in post["ts"]})
    tidsstamplar: dict[str, dict | None] = {tid: None for tid in alla_tider}

    for runda in (1, 2, 3):
        att_hamta = [tid for tid in alla_tider if tidsstamplar[tid] is None]
        if not att_hamta:
            break
        if runda > 1:
            log.info("Tidsserie-metadata, extrapass %d: %d anrop återstår.",
                     runda, len(att_hamta))
            time.sleep(10)

        def _hamta(tid):
            return tid, _hamta_tidsserie_meta(luft_api, tid)

        with ThreadPoolExecutor(max_workers=8) as pool:
            for tid, meta in pool.map(_hamta, att_hamta):
                tidsstamplar[tid] = meta
        kvar = sum(1 for tid in alla_tider if tidsstamplar[tid] is None)
        log.info("Tidsserie-metadata pass %d: %d klar, %d misslyckade.",
                 runda, len(alla_tider) - kvar, kvar)

    # Kopiera metadata tillbaka till stationernas tidsserier
    for post in stationer.values():
        for tid in post["ts"]:
            post["ts"][tid] = tidsstamplar.get(tid)

    nu_ms = datetime.now(timezone.utc).timestamp() * 1000
    alder_grans_ms = max_serie_alder_dagar * 24 * 3600 * 1000

    resultat = []
    for sid, post in stationer.items():
        # Välj serien med senaste lastValue (hoppar över döda/olästa serier)
        basta_tid, basta = None, None
        for tid, meta in post["ts"].items():
            if not meta or meta.get("last") is None:
                continue
            if meta.get("fenomen") not in ("", PM25_FENOMEN_ID):
                continue
            if basta is None or meta["last"] > basta["last"]:
                basta_tid, basta = tid, meta

        if basta is None or basta_tid is None:
            log.info("%s: ingen läsbar PM2.5-serie — stationen hoppas över.",
                     post["namn"])
            continue
        if nu_ms - basta["last"] > alder_grans_ms:
            log.info(
                "%s: senaste värdet %s är äldre än %d dagar — stationen "
                "rapporterar inte aktivt och hoppas över.",
                post["namn"], _datum_lokal(basta["last"]), max_serie_alder_dagar,
            )
            continue
        if basta.get("first") is None:
            continue

        resultat.append(Pm25Station(
            station_id=sid,
            namn=post["namn"],
            klass=post["klass"],
            lat=post["lat"],
            lon=post["lon"],
            tidsserie_id=int(basta_tid),
            serie_start=_datum_lokal(basta["first"]),
            serie_slut=_datum_lokal(basta["last"]),
        ))

    log.info("%d av %d PM2.5-stationer har en aktivt rapporterande serie.",
             len(resultat), len(stationer))
    return resultat


# ----------------------------------------------------------------
# Upptäckt: väderstationer
# ----------------------------------------------------------------

def hamta_vaderstationer(vader_api: str, parametrar: dict) -> list[Vaderstation]:
    """
    Hämtar SMHI-stationer som mäter ALLA parametrar i `parametrar`
    (t.ex. Lufttemperatur=1, ... Nederbördsmängd=7) och är aktiva.

    Snittet av stationslistorna per parameter; data_fran blir det SENASTE
    av parametrarnas startdatum, dvs. första dag alla parametrar finns.
    """
    stationer: dict[int, dict] = {}
    for namn, nummer in parametrar.items():
        svar = _api_get(f"{vader_api}/parameter/{nummer}.json", timeout=(15, 180))
        data = svar.json()
        for s in data["station"]:
            sid = int(s["id"])
            post = stationer.setdefault(
                sid,
                {"namn": s["name"], "lat": float(s["latitude"]),
                 "lon": float(s["longitude"]), "fra": [], "aktiva": []},
            )
            # 'from' är epoch-ms; vissa stationer saknar värde
            fran = s.get("from")
            post["fra"].append(fran if fran is not None else -2_147_483_648_000)
            post["aktiva"].append(bool(s.get("active")))
        log.info("Parameter %s (%s): %d stationer listade.", nummer, namn, len(data["station"]))
        time.sleep(0.3)

    resultat = []
    for sid, post in stationer.items():
        if len(post["fra"]) != len(parametrar):
            continue  # mäter inte alla parametrar
        if not all(post["aktiva"]):
            continue  # någon parameter inaktiv
        fran_ms = max(post["fra"])
        resultat.append(Vaderstation(
            station_id=sid,
            namn=post["namn"],
            lat=post["lat"],
            lon=post["lon"],
            data_fran=pd.Timestamp(fran_ms, unit="ms", utc=True).tz_convert(TIDZON),
            aktiv=True,
        ))

    log.info("%d aktiva väderstationer mäter alla %d parametrar.",
             len(resultat), len(parametrar))
    return resultat


# ----------------------------------------------------------------
# Parning
# ----------------------------------------------------------------

def para_stationer(
    pm25_stationer: list[Pm25Station],
    vaderstationer: list[Vaderstation],
    max_km: float,
    min_traningsmanader: int,
    validering_manader: int,
    train_rullande_ar: int,
) -> list[StationPar]:
    """
    Parar varje PM2.5-station med sin närmaste väderstation inom max_km.

    Filter, i ordning:
      1. Minst min_traningsmanader hela månader träningsdata efter att
         validering_manader reserverats (seriens start jämförs med
         idag minus båda marginerna).
      2. Väderstationen måste ha data (data_fran) sedan träningsfönstrets
         start = max(seriens start, idag - train_rullande_ar).
      3. Närmaste kvalificerade väderstation inom max_km.

    En väderstation kan väljas av flera PM2.5-stationer (medvetet).
    """
    idag = pd.Timestamp.now(tz=TIDZON).normalize()
    krav_start = idag - pd.DateOffset(
        months=min_traningsmanader + validering_manader,
    )
    rullande_start = idag - pd.DateOffset(years=train_rullande_ar)

    par = []
    for st in pm25_stationer:
        serie_start_ts = pd.Timestamp(st.serie_start, tz=TIDZON)

        # 1. Träningsunderlaget måste räcka (min-månadsregeln)
        if serie_start_ts > krav_start:
            log.info(
                "%s: serien startade %s — mindre än %d månader träning "
                "efter valideringens %d månader. Hoppas över.",
                st.namn, st.serie_start, min_traningsmanader, validering_manader,
            )
            continue

        # 2. Träningsfönstrets start (rullande kap eller seriens start)
        fonster_start = max(serie_start_ts, rullande_start)

        basta = None
        for vs in vaderstationer:
            if vs.data_fran > fonster_start:
                continue  # saknar data för delar av träningsfönstret
            d = haversine_km(st.lat, st.lon, vs.lat, vs.lon)
            if basta is None or d < basta[1]:
                basta = (vs, d)

        if basta is None or basta[1] > max_km:
            log.info(
                "%s: ingen kvalificerad väderstation inom %.0f km — hoppas över.",
                st.namn, max_km,
            )
            continue

        vs, d = basta
        log.info(
            "Par: %s (%s, serie %s, start %s) + %s [id %d] — %.1f km",
            st.namn, st.klass, st.tidsserie_id, st.serie_start,
            vs.namn, vs.station_id, d,
        )
        par.append(StationPar(
            pm25_station_id=st.station_id,
            pm25_namn=st.namn,
            pm25_klass=st.klass,
            pm25_lat=st.lat,
            pm25_lon=st.lon,
            tidsserie_id=st.tidsserie_id,
            serie_start=st.serie_start,
            serie_slut=st.serie_slut,
            vaderstation_id=vs.station_id,
            vaderstation_namn=vs.namn,
            vaderstation_lat=vs.lat,
            vaderstation_lon=vs.lon,
            avstand_km=round(d, 3),
        ))

    par.sort(key=lambda p: p.avstand_km)
    log.info("%d stationpar inom %.0f km klarade alla filter.", len(par), max_km)
    return par


# ----------------------------------------------------------------
# Cache (CSV)
# ----------------------------------------------------------------

def skriv_stationpar_cache(sokvag: Path, par: list[StationPar]) -> None:
    """Skriver parslistan som CSV — människoläsbar innan en körning."""
    rader = []
    for p in par:
        rader.append({
            "pm25_station_id": p.pm25_station_id,
            "pm25_namn": p.pm25_namn,
            "pm25_klass": p.pm25_klass,
            "pm25_lat": p.pm25_lat,
            "pm25_lon": p.pm25_lon,
            "tidsserie_id": p.tidsserie_id,
            "serie_start": p.serie_start,
            "serie_slut": p.serie_slut,
            "vaderstation_id": p.vaderstation_id,
            "vaderstation_namn": p.vaderstation_namn,
            "vaderstation_lat": p.vaderstation_lat,
            "vaderstation_lon": p.vaderstation_lon,
            "avstand_km": p.avstand_km,
            "beraknad_tid": datetime.now().isoformat(timespec="seconds"),
        })
    sokvag = Path(sokvag)
    sokvag.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rader, columns=CACHE_KOLUMNER).to_csv(sokvag, index=False)
    log.info("Skrev %d stationpar till cachen: %s", len(par), sokvag)


def las_stationpar_cache(sokvag: Path) -> tuple[list[StationPar], pd.Timestamp]:
    """
    Läser parslistan ur cachen. Returnerar (par, beraknad_tid).

    Kastar ValueError om filen saknar förväntade kolumner.
    """
    sokvag = Path(sokvag)
    df = pd.read_csv(sokvag)
    saknade = [k for k in CACHE_KOLUMNER if k not in df.columns]
    if saknade:
        raise ValueError(f"Stationparscachen {sokvag} saknar kolumner: {saknade}")

    par = [
        StationPar(
            pm25_station_id=int(rad["pm25_station_id"]),
            pm25_namn=str(rad["pm25_namn"]),
            pm25_klass=str(rad["pm25_klass"]),
            pm25_lat=float(rad["pm25_lat"]),
            pm25_lon=float(rad["pm25_lon"]),
            tidsserie_id=int(rad["tidsserie_id"]),
            serie_start=str(rad["serie_start"]),
            serie_slut=str(rad["serie_slut"]),
            vaderstation_id=int(rad["vaderstation_id"]),
            vaderstation_namn=str(rad["vaderstation_namn"]),
            vaderstation_lat=float(rad["vaderstation_lat"]),
            vaderstation_lon=float(rad["vaderstation_lon"]),
            avstand_km=float(rad["avstand_km"]),
        )
        for _, rad in df.iterrows()
    ]
    beraknad = pd.Timestamp(df["beraknad_tid"].iloc[-1])
    return par, beraknad


def _cache_is_fresh(sokvag: Path, cache_giltig_dagar: int) -> bool:
    if not sokvag.is_file():
        return False
    try:
        _, beraknad = las_stationpar_cache(sokvag)
    except Exception as e:
        log.warning(
            "Stationparscachen %s är oläslig (%s) — kör om upptäckten.",
            sokvag, e,
        )
        return False
    alder = pd.Timestamp.now() - beraknad
    return alder <= pd.Timedelta(days=cache_giltig_dagar)


def hamta_stationpar(bas) -> list[StationPar]:
    """
    Huvudingång: returnerar parslistan ur cachen om den är färsk,
    annars körs en full upptäckt och cachen skrivs om.

    Vid nätverksfel under upptäckt faller funktionen tillbaka på en
    äldre cache om en sådan finns — live-pipelinen ska inte stå helt
    still bara för att ett SMHI-API:hostar en kväll.

    `bas` är en BasConfig från src.config.ladda_baskonfig.
    """
    sokvag = Path(bas.projekt_rot) / bas.stationpar_cache

    if _cache_is_fresh(sokvag, bas.cache_giltig_dagar):
        par, beraknad = las_stationpar_cache(sokvag)
        log.info("Läser %d stationpar ur färsk cache (%s, beräknad %s).",
                 len(par), sokvag, beraknad)
        return par

    log.info("Stationparcachen saknas eller är äldre än %d dagar — kör "
             "full upptäckt mot API:erna (kan ta några minuter).",
             bas.cache_giltig_dagar)
    try:
        pm25_stationer = hamta_pm25_stationer(bas.luft_api, bas.max_serie_alder_dagar)
        vaderstationer = hamta_vaderstationer(bas.vader_api, bas.vader_parametrar)
        par = para_stationer(
            pm25_stationer,
            vaderstationer,
            max_km=bas.max_km,
            min_traningsmanader=bas.min_traningsmanader,
            validering_manader=bas.validering_manader,
            train_rullande_ar=bas.train_rullande_ar,
        )
        if not par:
            raise RuntimeError("Upptäckten hittade inga stationpar.")
        skriv_stationpar_cache(sokvag, par)
        return par
    except Exception as e:
        if sokvag.is_file():
            par, beraknad = las_stationpar_cache(sokvag)
            log.warning(
                "Upptäkten misslyckades (%s) — använder den gamla cachen "
                "från %s med %d par.",
                e, beraknad, len(par),
            )
            return par
        raise
