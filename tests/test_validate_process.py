"""Tester för src/validate_process.py: mönsterstatistik för saknad data
(plan 04) — find_nan_gaps + sammanstall_bortfall. Fixturerna bygger på
dokumenterade mönster i Malmö-data: 12-h nederbördsblock (SMHI-refuserad
hel dag) och ett 17-h pm25-gap."""

import pandas as pd

from src.validate_process import find_nan_gaps, sammanstall_bortfall


def _df_med_gaps(n=48, start="2026-10-01 00:00"):
    """n hela timmar med två mönster:
    - Nederbördsmängd: ett 12-timmarsblock (rad 10–21)
    - pm25: ett 17-timmarsgap (rad 25–41)
    Alla andra värden är kompletta."""
    tid = pd.date_range(start, periods=n, freq="h")
    df = pd.DataFrame(
        {
            "tid": tid,
            "pm25": 5.0,
            "Nederbördsmängd": 0.2,
            "Lufttemperatur": 8.0,
        }
    )
    df.loc[10:21, "Nederbördsmängd"] = pd.NA
    df.loc[25:41, "pm25"] = pd.NA
    return df


def test_find_nan_gaps_hittar_blocket():
    """Ett 12-timmarsblock ger exakt en gap-rad med rätt start/slut/längd."""
    df = _df_med_gaps()
    tid = df["tid"]

    gaps = find_nan_gaps(df, "Nederbördsmängd")

    assert len(gaps) == 1
    gap = gaps.iloc[0]
    assert gap["start"] == tid[10]
    assert gap["end"] == tid[21]
    assert gap["hours"] == 12


def test_find_nan_gaps_utan_nan_ger_tom_tabell():
    df = _df_med_gaps()
    assert find_nan_gaps(df, "Lufttemperatur").empty


def test_sammanstall_bortfall_langsta_gap_och_antal():
    """Längsta gap och antal gap >= horisont per kolumn — pm25:s 17 h
    och nederbördens 12 h, den kompletta kolumnen 0."""
    df = _df_med_gaps()

    kolumnstatistik, gaprader = sammanstall_bortfall(
        df, ["pm25", "Nederbördsmängd", "Lufttemperatur"], horisont=6,
    )

    stat = kolumnstatistik.set_index("kolumn")

    assert stat.loc["pm25", "saknade"] == 17
    assert stat.loc["pm25", "langsta_gap_timmar"] == 17
    assert stat.loc["pm25", "antal_gap_over_horisont"] == 1

    assert stat.loc["Nederbördsmängd", "langsta_gap_timmar"] == 12
    assert stat.loc["Nederbördsmängd", "antal_gap_over_horisont"] == 1

    # Komplett kolumn: noll överallt — men raden finns, så datakontrollen
    # får samma kolumner vid varje körning.
    assert stat.loc["Lufttemperatur", "saknade"] == 0
    assert stat.loc["Lufttemperatur", "langsta_gap_timmar"] == 0
    assert stat.loc["Lufttemperatur", "antal_gap_over_horisont"] == 0

    # Båda mönstren är >= 6 h => en rad var i gap-tabellen, med månad.
    assert len(gaprader) == 2
    assert set(gaprader["kolumn"]) == {"pm25", "Nederbördsmängd"}
    assert (gaprader["langd_timmar"] == 17).sum() == 1
    assert (gaprader["langd_timmar"] == 12).sum() == 1
    assert (gaprader["manad"] == 10).all()  # oktober


def test_sammanstall_bortfall_kort_gap_far_ingen_rad_men_raknas():
    """Ett 3-h gap (< min_gap_timmar) syns inte i gap-tabellen men blir
    korrekt längsta-gap; under horisonten räknas det inte som långt."""
    # Serie utan de inbyggda mönstren (n=10 ger inga rader för dem)
    df = _df_med_gaps(n=10)
    df.loc[2:4, "pm25"] = pd.NA

    kolumnstatistik, gaprader = sammanstall_bortfall(
        df, ["pm25"], horisont=6,
    )

    stat = kolumnstatistik.iloc[0]
    assert stat["langsta_gap_timmar"] == 3
    assert stat["antal_gap_over_horisont"] == 0
    assert stat["saknade"] == 3
    assert gaprader.empty

    # I fullserien med 17-h gapet kvar dominerar det långa gapet
    # statistiken, och det korta gapet undertrycks bara ur gap-TABELLEN
    # (alfabetiskt: minst 6 h för en egen rad), inte ur statistiken.
    df_full = _df_med_gaps()
    df_full.loc[5:7, "pm25"] = pd.NA

    stat_full, gaprader_full = sammanstall_bortfall(
        df_full, ["pm25"], horisont=6,
    )

    assert stat_full.iloc[0]["langsta_gap_timmar"] == 17
    assert len(gaprader_full) == 1
    assert gaprader_full.iloc[0]["langd_timmar"] == 17


def test_sammanstall_bortfall_gap_i_borjan_av_serien():
    """Ett gap som börjar på rad 0 (t.ex. sensor nere vid periodstart)
    räknas med start = seriens första timme."""
    # n=10: de inbyggda mönstren (rad 10+, 25+) hamnar utanför serien
    df = _df_med_gaps(n=10)
    df.loc[0:1, "Nederbördsmängd"] = pd.NA

    kolumnstatistik, gaprader = sammanstall_bortfall(
        df, ["Nederbördsmängd"], horisont=6,
    )

    assert kolumnstatistik.iloc[0]["langsta_gap_timmar"] == 2
    assert len(gaprader) == 0  # 2 h < 6 h => ingen egen rad
    # find_nan_gaps fångar start-fallet:
    gaps = find_nan_gaps(df, "Nederbördsmängd")
    assert len(gaps) == 1
    assert gaps.iloc[0]["start"] == df["tid"][0]
    assert gaps.iloc[0]["end"] == df["tid"][1]


def test_sammanstall_bortfall_utan_gaps_overhuvudtaget():
    """Ett helt komplett dataset: statistikrader med nollor, tom
    gap-tabell — datakontrollen får sina kolumner, bortfallsanalysen
    skriver ingenting."""
    # En serie utan några av de inbyggda mönstren (index utanför serien)
    df = _df_med_gaps(n=24)  # pm25-gapet ligger rad 25–41, dvs. utanför

    kolumnstatistik, gaprader = sammanstall_bortfall(
        df, ["pm25", "Nederbördsmängd", "Lufttemperatur"], horisont=6,
    )

    # nederbördsblocket (rad 10–21) finns kvar i den här serien — fyll
    # i det igen så att datasetet är helt komplett.
    df.loc[10:21, "Nederbördsmängd"] = 0.5

    kolumnstatistik, gaprader = sammanstall_bortfall(
        df, ["pm25", "Nederbördsmängd", "Lufttemperatur"], horisont=6,
    )

    assert len(kolumnstatistik) == 3
    assert (kolumnstatistik["langsta_gap_timmar"] == 0).all()
    assert (kolumnstatistik["antal_gap_over_horisont"] == 0).all()
    assert (kolumnstatistik["saknade"] == 0).all()
    assert gaprader.empty
