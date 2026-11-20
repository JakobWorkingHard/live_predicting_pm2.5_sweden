from dataclasses import dataclass, field
from pathlib import Path
import tomllib
import pandas as pd


@dataclass(frozen=True)
class ProjectConfig:
    # Mappar
    raw_dir: Path
    merged_dir: Path
    validated_dir: Path
    processed_dir: Path
    models_dir: Path
    champion_log: Path
    datakontroll_log: Path
    bortfallsanalys_log: Path   # plan 04: gap-rader (mönstret, inte omfånget)

    # Fullständiga sökvägar till specifika filer
    train_pm25: Path
    train_vader: Path
    train_merged: Path
    train_validated: Path
    train_processed: Path

    # Valideringsdata
    val_pm25: Path
    val_vader: Path
    val_merged: Path
    val_validated: Path
    val_processed: Path

    # Förskjuten modellinje (måltidsväder)
    train_processed_forskjuten: Path
    val_processed_forskjuten: Path
    forskjuten_log: Path
    jamfordelse_log: Path
    forskjuten_vaderkolumner: list
    forskjuten_antal_dagar_sedan: int

    # Nisch-modellinjen (hög-PM, plan 01)
    nisch_log: Path

    
    # API:er
    luft_api: str
    vader_api: str

    # Modell
    horisont: int
    features: list
    candidate_selection_metric: str
    champion_selection_metric: str
    random_state: int
    target_column: str
    antal_dagar_sedan: int

    # Tidsperioder
    train_start: str
    train_slut: str
    validering_start: str
    validering_slut: str

    # PM2.5 Station
    pm25_station: str
    pm25_tidsserie_id: int
    pm25_lat: float
    pm25_lon: float

    # Väderstation
    vaderstation: str
    vaderstation_id: int
    vader_lat: float
    vader_lon: float
    vader_parametrar: dict

    # Live-prediktioner
    prognos_dir: Path
    prediktioner_log: Path
    skippad_log: Path   # plan 04: en rad per hoppad live-körning/linje med orsak
    prognos_url: str           # färdig formaterad: väderstationens lon/lat insatta
    prognos_arkiv_timmar: int   # ARKIV, EJ PREDIKTION: trimmad arkiv-CSV:n i prognos_dir
    prognos_bibehall_dagar: int    # arkivstädning: prognosfiler äldre än så många dagar tas bort
    pm25_arkiv_bibehall_dagar: int # arkivstädning: pm25-hämtningar äldre än så många dagar tas bort
    max_alder_timmar: int
    idempotent: bool
    prognos_parametrar: dict

    # Hyperparametersökning (plan 05, beslut 2026-11-15). Defaults
    # nedan motsvarar "sökning avstängd" – ofarligt om sektionen saknas
    # i config.toml.
    hparam_aktiverad: bool = False
    hparam_n_splits: int = 4
    hparam_lambda_high_pm: float = 0.0
    hparam_svr_max_rader: int = 0
    hparam_grids: dict = field(default_factory=dict)

    # Nisch-modellinjen (plan 01): regel för gränsen för höga PM2.5.
    # Default är ofarlig om sektionen [model.high_pm] saknas.
    high_pm_threshold_rule: str = "pm25_p90_training"

    # Kompositscore (plan 02): vikter + High-PM-underlagskrav som styr
    # kandidatval och championduell. Default = metrics.DEFAULT_COMPOSITE_VIKTER
    # (ofarligt om sektionen [model.composite] saknas i config.toml).
    composite_vikter: dict = field(default_factory=lambda: {
        "w_mae": 0.5, "w_high_mae": 0.35, "w_high_bias": 0.15,
        "min_antal_high": 10,
    })

    # Låst test-snapshot (plan 02): data/test/last används ENBART för
    # oberoende rapporterande utvärdering — aldrig för modellval.
    test_last_dir: Path | None = None

    # Testdata – sökvägar skapas när testperioden har beräknats
    test_pm25: Path | None = None
    test_vader: Path | None = None
    test_merged: Path | None = None
    test_validated: Path | None = None
    test_processed: Path | None = None


@dataclass(frozen=True)
class BasConfig:
    """
    Den globala (stationslösa) konfigurationen ur config.toml.

    Laddas ENBART från TOML-filen — inga API-anrop sker — och används
    sedan av bygg_station_config() för att bygga en ProjectConfig per
    stationpar. Sökvägar som ska vara unika per par lagras som mallar
    med {}-platshållare och formateras med paretets värden.
    """
    projekt_rot: Path

    # Baskataloger
    raw_dir: Path
    merged_dir: Path
    validated_dir: Path
    processed_dir: Path
    models_rot_dir: Path    # rot; per par används models_rot_dir / <tidsserie_id>

    # Sökvägsmallar per par ({pm25_tidsserie_id} bland platshållarna)
    test_last_mall: str
    champion_log_mall: str
    forskjuten_log_mall: str
    nisch_log_mall: str
    datakontroll_log_mall: str
    bortfallsanalys_log_mall: str    # plan 04: gap-logg per par
    jamfordelse_log_mall: str
    prognos_dir_mall: str
    prediktioner_log_mall: str
    skippad_log_mall: str             # plan 04: skipp-logg per par
    prognos_url_mall: str   # {LON}/{LAT} för väderstationens koordinater

    # Filnamnsmallar ([paths.filenames])
    filenames: dict

    # API:er
    luft_api: str
    vader_api: str

    # Modell
    horisont: int
    features: list
    candidate_selection_metric: str
    champion_selection_metric: str
    random_state: int
    target_column: str
    antal_dagar_sedan: int
    forskjuten_vaderkolumner: list
    forskjuten_antal_dagar_sedan: int
    high_pm_threshold_rule: str
    composite_vikter: dict
    hparam_aktiverad: bool
    hparam_n_splits: int
    hparam_lambda_high_pm: float
    hparam_svr_max_rader: int
    hparam_grids: dict

    # Tid & stationpar
    train_rullande_ar: int     # träningen börjar som senast så här många år tillbaka
    validering_manader: int
    max_km: float              # parningsradie PM2.5 <-> väderstation
    max_serie_alder_dagar: int  # PM2.5-serien måste ha rapporterat inom detta
    min_traningsmanader: int   # minst så många månader träning efter valideringen
    stationpar_cache: str     # relativ sökväg till CSV-cachen
    cache_giltig_dagar: int

    # Väderparametrar (gemensamma för alla stationer, [station].parametrar)
    vader_parametrar: dict

    # Live
    prognos_arkiv_timmar: int
    prognos_bibehall_dagar: int
    pm25_arkiv_bibehall_dagar: int
    max_alder_timmar: int
    idempotent: bool
    prognos_parametrar: dict


def ladda_baskonfig(PATH_CONFIG_TOML: Path, PATH_PROJECT_ROOT: Path) -> BasConfig:
    """
    Läser config.toml till en BasConfig. Inga API-anrop — de sker först
    per par i bygg_station_config / när main.py kör.
    """
    with open(PATH_CONFIG_TOML, "rb") as f:
        raw = tomllib.load(f)

    p = raw["paths"]
    fn = raw["paths"]["filenames"]
    api = raw["api"]
    mod = raw["model"]
    fork = raw["model"]["forskjuten"]
    hogpm = raw["model"].get("high_pm", {})
    komp = raw["model"].get("composite", {})
    sok = raw["model"].get("hparam_search", {})
    t = raw["time"]
    station = raw["station"]
    stations = raw.get("stations", {})
    live = raw["live"]

    # Baskataloger
    raw_dir = PATH_PROJECT_ROOT / p["raw_dir"]
    merged_dir = PATH_PROJECT_ROOT / p["merged_dir"]
    validated_dir = PATH_PROJECT_ROOT / p["validated_dir"]
    processed_dir = PATH_PROJECT_ROOT / p["processed_dir"]
    models_rot_dir = PATH_PROJECT_ROOT / p["models_dir"]

    # Skapa datakatalogerna (inkl. data/) om de saknas – t.ex. vid fresh
    # clone, eftersom data/ är gitignored. Sker INNAN något API-anrop.
    for katalog in (raw_dir, merged_dir, validated_dir, processed_dir):
        katalog.mkdir(parents=True, exist_ok=True)

    prognos_parametrar = live["prognos_parametrar"]
    vader_parametrar = station["parametrar"]

    # Sanity: prognosparametrarna måste mappa samma väderkolumner som
    # träningen använder, annars glider live och träning isär.
    for kalla, kolumner in (
        ("[model.forskjuten].vaderkolumner", fork["vaderkolumner"]),
        ("[station].parametrar", list(vader_parametrar)),
    ):
        if sorted(prognos_parametrar) != sorted(kolumner):
            raise ValueError(
                f"[live].prognos_parametrar matchar inte {kalla}: "
                f"{sorted(prognos_parametrar)} != {sorted(kolumner)}"
            )

    # Defaults för [stations] — ofarliga om sektionen saknas.
    max_km = float(stations.get("max_km", 10))
    max_serie_alder_dagar = int(stations.get("max_serie_alder_dagar", 14))
    min_traningsmanader = int(stations.get("min_traningsmanader", 24))
    stationpar_cache = stations.get("stationpar_cache", "data/stationer/stationspar_v1.csv")
    cache_giltig_dagar = int(stations.get("cache_giltig_dagar", 7))

    return BasConfig(
        projekt_rot=PATH_PROJECT_ROOT,

        raw_dir=raw_dir,
        merged_dir=merged_dir,
        validated_dir=validated_dir,
        processed_dir=processed_dir,
        models_rot_dir=models_rot_dir,

        test_last_mall=p.get("test_last_dir", "data/test/last_{pm25_tidsserie_id}"),
        champion_log_mall=p["champion_log"],
        forskjuten_log_mall=p["forskjuten_log"],
        nisch_log_mall=p["nisch_log"],
        datakontroll_log_mall=p["datakontroll_log"],
        bortfallsanalys_log_mall=p.get(
            "bortfallsanalys_log",
            "logs/bortfallsanalys_{pm25_tidsserie_id}.csv",
        ),
        jamfordelse_log_mall=p["jamfordelse_log"],
        prognos_dir_mall=live["prognos_dir"],
        prediktioner_log_mall=live["prediktioner_log"],
        skippad_log_mall=live.get(
            "skippad_log",
            "logs/live_skippad_{pm25_tidsserie_id}.csv",
        ),
        prognos_url_mall=live["prognos_url"],

        filenames=dict(fn),

        luft_api=api["luft"],
        vader_api=api["vader"],

        horisont=mod["horisont"],
        features=mod["features"]["columns"],
        candidate_selection_metric=mod["candidate_selection_metric"],
        champion_selection_metric=mod["champion_selection_metric"],
        random_state=mod["random_state"],
        target_column=mod["target_column"],
        antal_dagar_sedan=mod["antal_dagar_sedan"],
        forskjuten_vaderkolumner=fork["vaderkolumner"],
        forskjuten_antal_dagar_sedan=fork["antal_dagar_sedan"],
        high_pm_threshold_rule=str(hogpm.get("threshold_rule", "pm25_p90_training")),
        composite_vikter={
            "w_mae": float(komp.get("w_mae", 0.5)),
            "w_high_mae": float(komp.get("w_high_mae", 0.35)),
            "w_high_bias": float(komp.get("w_high_bias", 0.15)),
            "min_antal_high": int(komp.get("min_antal_high", 10)),
        },
        hparam_aktiverad=bool(sok.get("aktiverad", False)),
        hparam_n_splits=int(sok.get("n_splits", 4)),
        hparam_lambda_high_pm=float(sok.get("lambda_high_pm", 0.0)),
        hparam_svr_max_rader=int(sok.get("svr_max_rader", 0)),
        hparam_grids=dict(sok.get("grid", {})),

        train_rullande_ar=int(t.get("train_rullande_ar", 4)),
        validering_manader=int(t["validering_manader"]),
        max_km=max_km,
        max_serie_alder_dagar=max_serie_alder_dagar,
        min_traningsmanader=min_traningsmanader,
        stationpar_cache=stationpar_cache,
        cache_giltig_dagar=cache_giltig_dagar,

        vader_parametrar=dict(vader_parametrar),

        prognos_arkiv_timmar=live["prognos_arkiv_timmar"],
        prognos_bibehall_dagar=live["prognos_bibehall_dagar"],
        pm25_arkiv_bibehall_dagar=live["pm25_arkiv_bibehall_dagar"],
        max_alder_timmar=live["max_alder_timmar"],
        idempotent=live["idempotent"],
        prognos_parametrar=dict(prognos_parametrar),
    )


def bygg_station_config(bas: BasConfig, par) -> ProjectConfig:
    """
    Bygger en komplett ProjectConfig för ETT stationpar (par: StationPar
    ur src.stations).

    Träningsstart (beslut 2026-11-15): rullande fönster — träningen
    börjar vid max(seriens första värde, idag - train_rullande_ar år).
    Serien är tjockare än fönstret kapas den alltså; är den yngre än
    fönstret börjar den vid sitt första värde.

    Hämtar senaste arkivtidpunkt för PARETS väderstation (API-anrop,
    5 parametrar) och beräknar tränings-/valideringsperioderna. Alla
    loggar, modellkataloger och prognosarkiv är namnrymdade per par
    via tidsserie-ID:t, så paren aldrig kan skriva över varandra.
    """
    from src.extracting_data import hamta_senaste_arkivtid

    # Rullande träningsstart: max(seriens start, idag - train_rullande_ar).
    idag = pd.Timestamp.now(tz="Europe/Stockholm").normalize()
    rullande_start = (
        idag - pd.DateOffset(years=bas.train_rullande_ar)
    ).strftime("%Y-%m-%d")
    train_start = max(par.serie_start, rullande_start)

    # Hämta senaste tillgängliga arkivtidpunkt för parets väderstation
    senaste_arkivtid = hamta_senaste_arkivtid(
        parametrar=bas.vader_parametrar,
        station_id=par.vaderstation_id,
        VADER_API=bas.vader_api,
    )

    # Beräkna dynamiska tränings- och valideringsperioder
    perioder = berakna_tranings_och_valideringsperiod(
        senaste_arkivtid=senaste_arkivtid,
        train_start=train_start,
        validering_manader=bas.validering_manader,
    )

    train_slut = perioder["train_slut"]
    validering_start = perioder["validering_start"]
    validering_slut = perioder["validering_slut"]

    logiska_datum = {
        "train_start": train_start,
        "train_slut": train_slut,
        "validering_start": validering_start,
        "validering_slut": validering_slut,
    }

    print(f"\n=== {par.pm25_namn} + {par.vaderstation_namn} "
          f"(tidsserie {par.tidsserie_id}) ===")
    print("Dynamiskt beräknade datum:")
    for namn, datum in logiska_datum.items():
        print(f"{namn}: {datum}")

    # Värden för filnamnsmallar (stationsnamn sanerade som tidigare)
    fmt = {
        "pm25_station": par.pm25_namn.replace(" ", "_"),
        "vaderstation": par.vaderstation_namn.replace(" ", "_"),
        "train_start": train_start,
        "train_slut": train_slut,
        "pm25_tidsserie_id": par.tidsserie_id,
        "vaderstation_id": par.vaderstation_id,
        "horisont": bas.horisont,
        "validering_start": validering_start,
        "validering_slut": validering_slut,
    }

    return ProjectConfig(
        # Mappar — modellfiler per par i egen underkatalog
        raw_dir=bas.raw_dir,
        merged_dir=bas.merged_dir,
        validated_dir=bas.validated_dir,
        processed_dir=bas.processed_dir,
        models_dir=bas.models_rot_dir / str(par.tidsserie_id),
        champion_log=bas.projekt_rot / bas.champion_log_mall.format(**fmt),
        datakontroll_log=bas.projekt_rot / bas.datakontroll_log_mall.format(**fmt),
        bortfallsanalys_log=bas.projekt_rot / bas.bortfallsanalys_log_mall.format(**fmt),

        # Fullständiga sökvägar till träningsdata
        train_pm25=bas.raw_dir / bas.filenames["train_pm25"].format(**fmt),
        train_vader=bas.raw_dir / bas.filenames["train_vader"].format(**fmt),
        train_merged=bas.merged_dir / bas.filenames["train_merged"].format(**fmt),
        train_validated=bas.validated_dir / bas.filenames["train_validated"].format(**fmt),
        train_processed=bas.processed_dir / bas.filenames["train_processed"].format(**fmt),

        # Fullständiga sökvägar till valideringsdata
        val_pm25=bas.raw_dir / bas.filenames["val_pm25"].format(**fmt),
        val_vader=bas.raw_dir / bas.filenames["val_vader"].format(**fmt),
        val_merged=bas.merged_dir / bas.filenames["val_merged"].format(**fmt),
        val_validated=bas.validated_dir / bas.filenames["val_validated"].format(**fmt),
        val_processed=bas.processed_dir / bas.filenames["val_processed"].format(**fmt),

        # Förskjuten modellinje (måltidsväder)
        train_processed_forskjuten=bas.processed_dir / bas.filenames["train_processed_forskjuten"].format(**fmt),
        val_processed_forskjuten=bas.processed_dir / bas.filenames["val_processed_forskjuten"].format(**fmt),
        forskjuten_log=bas.projekt_rot / bas.forskjuten_log_mall.format(**fmt),
        nisch_log=bas.projekt_rot / bas.nisch_log_mall.format(**fmt),
        jamfordelse_log=bas.projekt_rot / bas.jamfordelse_log_mall.format(**fmt),
        forskjuten_vaderkolumner=bas.forskjuten_vaderkolumner,
        forskjuten_antal_dagar_sedan=bas.forskjuten_antal_dagar_sedan,

        # API:er
        luft_api=bas.luft_api,
        vader_api=bas.vader_api,

        # Modell
        horisont=bas.horisont,
        features=bas.features,
        candidate_selection_metric=bas.candidate_selection_metric,
        champion_selection_metric=bas.champion_selection_metric,
        random_state=bas.random_state,
        target_column=bas.target_column,
        antal_dagar_sedan=bas.antal_dagar_sedan,

        # Tidsperioder
        train_start=train_start,
        train_slut=train_slut,
        validering_start=validering_start,
        validering_slut=validering_slut,

        # PM2.5 Station
        pm25_station=par.pm25_namn,
        pm25_tidsserie_id=par.tidsserie_id,
        pm25_lat=par.pm25_lat,
        pm25_lon=par.pm25_lon,

        # Väderstation
        vaderstation=par.vaderstation_namn,
        vaderstation_id=par.vaderstation_id,
        vader_lat=par.vaderstation_lat,
        vader_lon=par.vaderstation_lon,
        vader_parametrar=bas.vader_parametrar,

        # Live-prediktioner — arkiv och logg per par
        prognos_dir=bas.projekt_rot / bas.prognos_dir_mall.format(**fmt),
        prediktioner_log=bas.projekt_rot / bas.prediktioner_log_mall.format(**fmt),
        skippad_log=bas.projekt_rot / bas.skippad_log_mall.format(**fmt),
        prognos_url=bas.prognos_url_mall.format(
            LON=par.vaderstation_lon, LAT=par.vaderstation_lat,
        ),
        prognos_arkiv_timmar=bas.prognos_arkiv_timmar,
        prognos_bibehall_dagar=bas.prognos_bibehall_dagar,
        pm25_arkiv_bibehall_dagar=bas.pm25_arkiv_bibehall_dagar,
        max_alder_timmar=bas.max_alder_timmar,
        idempotent=bas.idempotent,
        prognos_parametrar=bas.prognos_parametrar,

        # Hyperparametersökning (plan 05)
        hparam_aktiverad=bas.hparam_aktiverad,
        hparam_n_splits=bas.hparam_n_splits,
        hparam_lambda_high_pm=bas.hparam_lambda_high_pm,
        hparam_svr_max_rader=bas.hparam_svr_max_rader,
        hparam_grids=bas.hparam_grids,

        # Nisch-modellinjen (plan 01)
        high_pm_threshold_rule=bas.high_pm_threshold_rule,

        # Kompositscore (plan 02)
        composite_vikter=dict(bas.composite_vikter),

        # Låst test-snapshot (plan 02) — per par
        test_last_dir=bas.projekt_rot / bas.test_last_mall.format(**fmt),
    )


def berakna_tranings_och_valideringsperiod(
    senaste_arkivtid: pd.Timestamp,
    train_start: str,
    validering_manader: int,
) -> dict[str, str]:
    """
    Beräknar dynamiska tränings- och valideringsperioder.

    Valideringen består av de `validering_manader` senaste fullständiga
    kalendermånaderna före den månad som latest_arcives tid tillhör.

    Träningen slutar dagen före valideringsperiodens start.
    """

    # Början av månaden för den senaste arkivtidpunkten
    aktuell_manad = senaste_arkivtid.normalize().replace(
        day=1,
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
    )

    # Valideringsperioden omfattar de föregående hela månaderna
    validering_slut = aktuell_manad - pd.Timedelta(days=1)
    validering_start = (
        aktuell_manad - pd.DateOffset(months=validering_manader)
    ).normalize()

    # Träningen slutar dagen före valideringen börjar
    train_slut = validering_start - pd.Timedelta(days=1)

    return {
        "train_start": train_start,
        "train_slut": train_slut.strftime("%Y-%m-%d"),
        "validering_start": validering_start.strftime("%Y-%m-%d"),
        "validering_slut": validering_slut.strftime("%Y-%m-%d"),
    }


def skapa_testvagar(
    cfg: ProjectConfig,
    test_start: pd.Timestamp,
    test_slut: pd.Timestamp,
) -> dict[str, Path]:
    """Skapar sökvägar för testfiler baserat på den dynamiska testperioden.

    Filnamnen namnrymds med tidsserie- och stations-ID — flera par delar
    rå-/merged-/processed-kataloger och samma testperiod, så utan
    ID:na skulle paren skriva över varandras filer."""

    test_start_str = test_start.strftime("%Y-%m-%d")
    test_slut_str = test_slut.strftime("%Y-%m-%d")
    stations_suffix = (
        f"pm25tidsserie_{cfg.pm25_tidsserie_id}"
        f"_vaderstationid_{cfg.vaderstation_id}"
    )

    return {
        "test_pm25": cfg.raw_dir / (
            f"test_{cfg.pm25_station.replace(' ', '_')}_"
            f"{test_start_str}_{test_slut_str}_"
            f"pm25tidsserie_{cfg.pm25_tidsserie_id}.csv"
        ),
        "test_vader": cfg.raw_dir / (
            f"test_{cfg.vaderstation.replace(' ', '_')}_"
            f"{test_start_str}_{test_slut_str}_"
            f"vaderstationid_{cfg.vaderstation_id}.csv"
        ),
        "test_merged": cfg.merged_dir / (
            f"test_merged_{test_start_str}_{test_slut_str}_"
            f"{stations_suffix}.csv"
        ),
        "test_validated": cfg.validated_dir / (
            f"test_validated_{test_start_str}_{test_slut_str}_"
            f"{stations_suffix}.csv"
        ),
        "test_processed": cfg.processed_dir / (
            f"test_processed_{test_start_str}_{test_slut_str}_"
            f"{stations_suffix}.csv"
        ),
        "test_processed_forskjuten": cfg.processed_dir / (
            f"test_processed_forskjuten_{test_start_str}_{test_slut_str}"
            f"_h{cfg.horisont}_{stations_suffix}.csv"
        ),
    }
