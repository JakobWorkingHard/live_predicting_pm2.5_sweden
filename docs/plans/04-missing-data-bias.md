# Plan 04: Missing data — keep the dropna policy, log it properly (batch AND live)

Part of the limitation-fix series for Predicting Air Quality
(PM2.5, Malmö Dalaplan). Scope decided with the team 2026-11-15:
the `dropna` batch policy is **fine** as long as missing data is
reliably *logged*. The genuinely open question is **live mode**: what
happens when data is missing at prediction time.

## Limitation / open question

- Batch: missing data is non-random (precipitation missing in whole
  day-blocks where SMHI rejected the measurements; PM2.5 gaps up to
  17 h; DST blocks). Team decision: keep `dropna`, ensure logging
  makes the pattern visible and trackable.
- Live: **what happens when input data is missing at prediction
  time?** Today the answer is: mostly silent skips, and in one case
  an unhandled crash (details below).

## Current state (verified 2026-11-15)

### Batch side
- `src/data_pipeline.py::build_processed_dataset()` drops rows with
  NaN in features/target and appends before/after counts plus per
  column missing counts to `logs/datakontroll.csv`
  (`append_datakontroll_log()`) — the *extent* is logged, but not the
  *pattern* (no gap lengths, no block structure).

### Live side (the real gap)
Verified behavior when data is missing at prediction time:
1. **Missing weather observation or PM2.5 at the current hour** →
   `bestam_ankarpunkt()` falls back to the latest *complete* hour.
   Fine — but if that hour is older than `[live].max_alder_timmar`
   (6 h) the whole live run is **skipped with only a log warning**;
   nothing is written to `prediktioner_live.csv`, so the follow-up
   cannot see how often/why live predictions were skipped.
2. **Missing forecast row for the måltid (T_obs + horisont)** →
   `bygg_live_rader()` **raises `ValueError`** — the live run propagates
   an exception instead of skipping gracefully.
3. Idempotency and per-champion-era trimming
   (`trim_prediktioner_log`) are unaffected by missing data, but a
   champion-era change also silently discards unrealized predictions
   (noted in decisions 2026-11-15).

## Goal

1. Batch: log the missing-data *pattern* (gaps, blocks, seasonality) —
   no change to the dropna policy itself.
2. Live: make every missing-data path **explicit, graceful and
   logged**, so the live system's reliability is measurable and a
   crash can never come from an absent forecast row.

## Implementation steps

### 04.1 Batch: missingness pattern log
- Extend `append_datakontroll_log()` (or sibling
  `sammanstall_bortfall()` writing `logs/bortfallsanalys.csv`):
  - per column: missing count/share, **longest consecutive gap**,
    number of gaps ≥ `horisont` hours;
  - per gap ≥ 6 h: start, end, duration, month — surfaces the
    day-block precipitation pattern and the 17-h PM2.5 gap type.
- Print a compact summary in the run log each pipeline run.
- Docs: summarize the pattern in `docs/data.md` (and, if the blocks
  turn out systematic, a sentence in docs/evaluation.md limitations).

### 04.2 Live: explicit skip log (`logs/live_skippad.csv`)
- New lightweight log appended by `kor_live_prediktion()` whenever a
  live run or a single line is skipped. Columns: `timestamp`,
  `t_obs` (if determined), `orsak` (enum: `ingen_ankarpunkt`,
  `for_gammal_ankarpunkt`, `t_obs_redan_predikterad`,
  `prognos_saknas_maltid`, `champion_saknas:<linje>`), `alinje`
  (whole-run vs. per-line), `pm25_alder_h`.
- Requirements:
  - every early-return/vakter path in `_kor_live_prediktion()` and
    `bygg_live_rader()` writes exactly one reason row;
  - the follow-up (`live.py --uppföljning`) prints a small skip
    summary (counts per reason over the last 7 days) next to the
    performance table, so "why are there so few predictions?" is
    answerable at a glance.

### 04.3 Live: fix the crash path
- In `bygg_live_rader()`: replace the `ValueError` for a missing
  prognos row with a logged skip (`orsak=prognos_saknas_maltid`) for
  the förskjuna line; the vanliga line can still be served, since it
  needs only T-slot observations. Currently the exception kills both
  lines — so: catch at the slot level and serve whichever lines' slots
  are complete.
- Same philosophy if any weather parameter in a slot fails
  `sanity_kolla_vader`: serve the lines that pass, log the lines that
  don't.

### 04.4 (No data-handling changes)
- Explicitly **not** doing: interpolation (banned, decisions
  2026-10-09/12), bounded forward-fill + indicator features (rejected
  by team 2026-11-15 — keep dropna), alternative precipitation
  sources. If missingness turns out to grow or to distort High-PM
  evaluation (plan 01), revisit then, with the 04.1 log as evidence.

## Affected files

- `src/io.py` — `append_datakontroll_log()` extension, new skip-log
  writer
- `src/validate_process.py` — gap/block statistics helper for 04.1
- `src/predict_pipeline.py` — reason instrumentation on all vakter,
  prognos-slot graceful skip, skip summary in `folj_uppa()`
- `live.py` — (no change; flag stays)
- `config.toml` / `src/config.py` — new log path
  `logs/live_skippad.csv`
- `tests/test_io.py`, `tests/test_predict_pipeline.py`,
  `tests/test_extracting_data.py`

## Tests

- Gap statistics: synthetic series with a 12-h precipitation block and
  a 17-h PM2.5 gap → correct longest gap, counts, gap rows with
  month labels.
- Every vakt path writes exactly one skip row with the right `orsak`;
  a normal run writes none.
- Missing prognos row for måltid: vanlig linje still predicted and
  logged; förskjuten linje skipped with `prognos_saknas_maltid`; no
  exception escapes `kor_live_prediktion()`.
- `folj_uppa()` skip summary counts match the log fixture.

## Acceptance criteria

- Every pipeline run updates `datakontroll`/`bortfallsanalys` with
  pattern-level stats; the day-block precipitation pattern is
  confirmed/refuted with numbers in `docs/data.md`.
- A missing forecast or observation in live mode can never crash the
  run; every skip is machine-readable and summarized in the
  follow-up.
- After a week of live runs, "prediktioner som gjorts" vs "körningar
  som hoppats över + varför" is a one-glance table.

## Risks & notes

- The skip log grows ~1 row per skipped live attempt — bounded by the
  already-throttled live frequency; no cleanup needed initially.
- Serving the vanliga line while the förskjuna line is skipped is a
  small behavioral change (today: all-or-nothing crash); document in
  docs/decisions.md.
- Plan 06's significance testing needs n: the skip log makes the
  effective sample size honest instead of silently shrinking.
