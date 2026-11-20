import pandas as pd

from src.logger import setup_logger

log = setup_logger(__name__)

# Gemensam källa för giltighetsintervall i väderdata (Lufttemperatur saknar
# regel, precis som i huvudpipelinens replace_invalid_values). Missbrukas
# av både tabellversionen (replace_invalid_values) och singelradsversionen
# (sanity_kolla_vader) i live-flödet.
VADER_INTERVALL = {
    "Vindriktning": (0, 360),
    "Vindhastighet": (0, None),
    "Relativ luftfuktighet": (0, 100),
    "Nederbördsmängd": (0, None),
}

def convert_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Konverterar kolumnen 'tid' till svensk timezone-aware datetime."""
    
    df = df.copy()

    df["tid"] = (
        pd.to_datetime(df["tid"], utc=True)
        .dt.tz_convert("Europe/Stockholm")
    )

    return df


def sort_by_time(df: pd.DataFrame) -> pd.DataFrame:
    """Sorterar observationerna kronologiskt efter 'tid'."""

    df = df.copy()

    return df.sort_values("tid").reset_index(drop=True)


def check_duplicates(df: pd.DataFrame) -> None:
    """Kontrollerar om det finns dubbla tidsstämplar."""

    duplicates = df["tid"].duplicated().sum()

    if duplicates > 0:
        log.warning(
            "Datasetet innehåller %s dubbla tidsstämplar.",
            duplicates,
        )

        raise ValueError(
            f"Datasetet innehåller {duplicates} dubbla tidsstämplar."
        )


def ensure_complete_timeline(df: pd.DataFrame) -> pd.DataFrame:
    """Säkerställer att det finns en rad för varje timme i tidsserien."""

    df = df.copy()

    full_timeline = pd.date_range(
        start=df["tid"].min(),
        end=df["tid"].max(),
        freq="h"
    )

    df = (
        df.set_index("tid")
        .reindex(full_timeline)
        .rename_axis("tid")
        .reset_index()
    )

    return df


def replace_invalid_values(df: pd.DataFrame) -> pd.DataFrame:
    """Ersätter uppenbart ogiltiga mätvärden med NaN."""
    df = df.copy()

    # PM2.5 kan inte vara negativt
    df.loc[df["pm25"] < 0, "pm25"] = pd.NA

    # I SMHI:s data betyder 0° vindstilla och 360° nord.
    # För sin/cos är 0° och 360° samma vinkel, så 360° sätts till 0°.
    # Vindstilla hanteras i add_wind_features (Vindhastighet == 0).
    df.loc[df["Vindriktning"] == 360, "Vindriktning"] = 0.0

    # Giltighetsintervall per väderkolumn — reglerna lever i VADER_INTERVALL
    for kolumn, (MIN, MAX) in VADER_INTERVALL.items():
        ogiltig = df[kolumn] < MIN
        if MAX is not None:
            ogiltig = ogiltig | (df[kolumn] > MAX)
        df.loc[ogiltig, kolumn] = pd.NA

    return df


def find_nan_gaps(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Hittar sammanhängande perioder med saknade värden i en kolumn.

    Används av sammanstall_bortfall (plan 04) för mönsterstatistiken
    över saknad data. Kräver kolumnen 'tid' — anropas på det VALIDERADE
    datat, där ensure_complete_timeline redan garanterat en rad per
    timme, så 'hours' är gapets längd i hela timmar.
    """

    missing = df[column].isna()

    # Varje gång vi går från NaN till icke-NaN eller tvärtom
    # börjar en ny grupp
    groups = missing.ne(missing.shift()).cumsum()

    gaps = (
        df[missing]
        .groupby(groups[missing])
        .agg(
            start=("tid", "first"),
            end=("tid", "last"),
            hours=("tid", "size")
        )
        .reset_index(drop=True)
    )

    return gaps


def sammanstall_bortfall(
    df: pd.DataFrame, kolumner: list, horisont: int, min_gap_timmar: int = 6,
):
    """
    Plan 04: sammanställer saknad datas MÖNSTER per kolumn — längsta
    sammanhängande gap och antal gap >= horisont, plus en rad per gap
    >= min_gap_timmar med start/slut/längd/månad. Gör dagsblocken
    (t.ex. SMHI-refuserad nederbörd) och långa pm25-gap synliga och
    spårbara i loggarna. dropna-policyn påverkas inte.

    MÅSTE anropas på det validerade datat FÖRE dropna — efter dropna
    är raderna borta och mönstret kan inte longer rekonstrueras (det är
    därför statistiken inte kan samlas in från de processade filerna).

    Parametrar
    ----------
    df : pd.DataFrame
        Validerat dataset med kolumnen 'tid' (komplett timaxel).
    kolumner : list
        Kolumner att undersöka, t.ex. features + target.
    horisont : int
        Prognoshorisonten i timmar — räknar gap som är minst så långa.
    min_gap_timmar : int
        Minsta längd för att ett gap ska få sin egen rad i gap-tabellen.

    Returnerar
    -------
    tuple
        (kolumnstatistik: pd.DataFrame med en rad per kolumn — kolumn,
        saknade, langsta_gap_timmar, antal_gap_over_horisont —
        gaprader: pd.DataFrame med en rad per gap >= min_gap_timmar —
        kolumn, start, slut, langd_timmar, manad).
    """
    stat_rader = []
    gap_rader = []

    for kolumn in kolumner:
        gaps = find_nan_gaps(df, kolumn)
        saknade = int(df[kolumn].isna().sum())

        langsta = int(gaps["hours"].max()) if not gaps.empty else 0
        antal_over = (
            int((gaps["hours"] >= horisont).sum()) if not gaps.empty else 0
        )
        stat_rader.append({
            "kolumn": kolumn,
            "saknade": saknade,
            "langsta_gap_timmar": langsta,
            "antal_gap_over_horisont": antal_over,
        })

        for _, gap in gaps[gaps["hours"] >= min_gap_timmar].iterrows():
            gap_rader.append({
                "kolumn": kolumn,
                "start": gap["start"],
                "slut": gap["end"],
                "langd_timmar": int(gap["hours"]),
                "manad": int(gap["start"].month),
            })

    return pd.DataFrame(stat_rader), pd.DataFrame(gap_rader)


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validerar tidsserie och mätvärden utan interpolation."""

    log.info("Startar datavalidering.")

    df = convert_datetime(df)
    df = sort_by_time(df)

    # Kontrollera att tidsstämplarna är unika
    check_duplicates(df)

    # Säkerställ en komplett timaxel
    df = ensure_complete_timeline(df)

    # Ersätt ogiltiga mätvärden med NaN
    df = replace_invalid_values(df)

    log.info(
        "Datavalidering slutförd. Antal rader: %s, antal kolumner: %s.",
        len(df),
        len(df.columns),
    )

    return df


# Används av predict pipeline
def sanity_kolla_vader(rad: pd.Series) -> pd.Series:
    """
    Intervallreglerna för en enskild rad väderdata (live-flödet):
    ogiltigt -> NaN, 360 grader -> 0.

    Exakt samma regler som replace_invalid_values tillämpar på hela
    tabeller — båda bygger på VADER_INTERVALL. `rad` förväntas vara
    en Series med väderkolumnerna; saknade värden (None/NaN) lämnas.
    """
    rad = rad.astype(float).copy()

    if rad["Vindriktning"] == 360:
        rad["Vindriktning"] = 0.0

    for kolumn, (MIN, MAX) in VADER_INTERVALL.items():
        if rad[kolumn] < MIN or (MAX is not None and rad[kolumn] > MAX):
            rad[kolumn] = pd.NA

    return rad