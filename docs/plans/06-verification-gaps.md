# Plan 06: Close the verification and analysis gaps

Part of the limitation-fix series for Predicting Air Quality
(PM2.5, Malmö Dalaplan). Source: report §4, §5 and §7 — several smaller
but distinct gaps, all confirmed in the codebase 2026-11-15.

## Limitations addressed

1. **Freshness gate only checks the vanliga champion.**
   `src/orchestrator.py` line ~26: "Endast den VANLIGA championen
   styr" — if the vanliga champion is fresh, the *entire* pipeline is
   skipped, so the förskjuten line is only ever retrained as a
   side-effect of the vanliga line going stale. (The report noted the
   guard "kontrollerar endast den vanliga championen"; the current
   design makes this deliberate but asymmetric.)
2. **High-PM analysis never run on test data.**
   The threshold analysis exists only on validation
   (docs/evaluation.md: "Motsvarande uppdelning efter koncentrationsnivå
   har inte genomförts för de slutliga championmodellerna på
   testdata"). Ironically, `src/metrics.py::jamfor_championer()` has a
   `high_pm25_threshold: float | None = None` parameter that
   `orchestrator.py` never passes → head-to-head reports no High-PM
   metrics.
3. **No statistical significance testing.**
   docs/evaluation.md ("Slutsatser som resultaten inte stödjer"):
   no test supports the vanlig-vs-förskjuten difference; only ~12 live
   predictions existed at report time; no significance machinery
   exists anywhere in the pipeline.
4. **Test data (`latest-months`) is unvalidated/preliminary** and can be
   revised by SMHI after the fact (report §2; docs/evaluation.md
   "Skillnader i datakvalitet"). Results on it can silently change.
5. **Live follow-up lacks High-PM metrics**
   (`sammanstall_live_uppfoljning` reports only MAE/RMSE/R²/Bias).

## Current state (verified 2026-11-15)

- `run_model_line(cfg, …, antal_dagar, log_path …)` is line-aware and
  uses `champion_is_fresh(log_path, antal_dagar, namn_suffix=…)` — the
  per-line mechanism exists; only the orchestrator's top-level gate is
  single-line.
- Champion thresholds: `y_train.quantile(0.90)` computed inside
  `train_and_evaluate_models`/`persistence_baseline` but never
  propagated to the orchestrator for `jamfor_championer`.
- Live follow-up: `src/predict_pipeline.py::folj_uppa()` +
  `src/metrics.py::sammanstall_live_uppfoljning()`.
- Logs available: `logs/datakontroll.csv`, champion logs,
  `logs/jamforelse_champion.csv`, `prognoser/prediktioner_live.csv`
  (now kept per champion era due to `trim_prediktioner_log`, NOTE
  "orealiserade prediktioner från gamla championen försvinner vid
  byte" — relevant for sample-size math).

## Goal

Make every produced number statistically and structurally trustworthy:
both lines retrain on their own schedule, High-PM analysis exists on
test and live data, differences between models are significance-tested,
and preliminary-data caveats are explicit.

## Implementation steps

### 6.1 Per-line freshness gates
- In `src/orchestrator.py`, compute freshness per line:
  `vanlig_farsk = champion_is_fresh(cfg.champion_log, …, vanlig_suffix)`
  and `forskjuten_farsk = champion_is_fresh(cfg.forskjuten_log,
  cfg.forskjuten_antal_dagar_sedan, …, forskjuten_suffix)`.
- Restructure so each stage runs independently: data fetch/processing
  when *either* line is stale; `run_model_line` per stale line;
  `jamfor_championer` when a förskjuten champion exists. The
  skip-branch prints both lines' status.
- Keep the champion-name suffix horizon protection intact (it guards
  against silent horizon changes).

### 6.2 High-PM metrics everywhere
- Compute the threshold once per run in the orchestrator phase
  (from processed training data, or persist it from
  `run_model_line` into the champion metadata `.json` — preferred,
  keeps a single source of truth) and pass it to
  `jamfor_championer(..., high_pm25_threshold=…)`.
- Add High-PM MAE/Bias columns to
  `sammanstall_live_uppfoljning()` (threshold from champion metadata
  per line; persistence baseline gets the same treatment).
- Add a one-off reporting step that scores the final champions on the
  locked test snapshot (from plan 02) with a High-PM split, closing the
  docs/evaluation.md gap "har inte genomförts … på testdata".

### 6.3 Significance testing in `src/metrics.py`
- New `signifikansprövning(errors_a, errors_b, …)`:
  - **Diebold–Mariano** for paired forecast errors (implement
    loss differential + autocorrelation-robust variance, since
    overlapping 6-hour horizons correlate neighboring errors), and/or
  - a paired moving-block bootstrap (simple, robust, easy to unit
    test): resample error *differences* in blocks ≥ horizon length,
    report the p-value and 95 % CI of the metric difference.
- Wire into: `compete_with_champion()` output (report the challenger's
  significance level before a switch is allowed to be called an
  "improvement" — optionally require p < α to *save a new champion*
  when the composite margin is tiny), and into the live follow-up
  report — a "does the förskjuna line exist in anything better than
  noise" verdict. Make it reusable by plan 01's high-PM adoption duel
  (is the niche advantage larger than noise on the filtered rows?).
- Minimum sample guards: no test when n < 30 (report "too little
  data" as today).

### 6.4 Preliminary-data flagging
- In `hamta_testperiod()`/`build_processed_dataset`, mark test data as
  preliminary: source column or a manifest (`data/test/kalla.json`)
  with dataset period ("latest-months" vs "corrected-archive").
- Optional refresh job: when SMHI's `corrected-archive` later covers
  the test window, re-download and re-report the locked-test metrics
  (report-only; selections never use test data after plan 02).
- Add the caveat to every test-based table in docs/evaluation.md
  unless/until corrected data replaced the window.

## Affected files

- `src/orchestrator.py` — per-line gates, threshold propagation
- `src/metrics.py` — significance functions, live high-PM summary
- `src/champion.py` — optional significance-gated switch rule
- `src/io.py` — champion metadata threshold field
- `src/extracting_data.py` — provenance marking, refresh job
- `src/predict_pipeline.py` — surfaced significance in uppföljning
- `docs/evaluation.md`, `docs/decisions.md`, `docs/data.md`
- `tests/test_champion.py`, `tests/test_metrics.py`,
  `tests/test_extracting_data.py`, `tests/test_predict_pipeline.py`

## Tests

- Per-line freshness: vanlig fresh + förskjuten stale → only the
  förskjuten line retrains; both stale → full pipeline; both fresh →
  skip branch with both statuses logged.
- `jamfor_championer` receives threshold and its output table includes
  High-PM columns (fixture with peaks above threshold).
- Significance helper: two identical error series → p ≈ 1; shifted
  series → p < 0.05; n < 30 → guard message; block bootstrap is
  deterministic per seed.
- Live summary with threshold: persistence row contains High-PM
  metrics computed per unique måltid (matching existing dedup rules).
- Provenance: test tables read from `latest-months` are marked as
  preliminary in the produced log/datakontroll row.

## Acceptance criteria

- Reports no longer contain implicit claims: every vanlig-vs-förskjuten
  / champion-vs-challenger difference states n, the test used and p
  (or is explicitly labeled "not yet tested — too few observations").
- Both model lines retrain on their own 7-day clocks.
- Head-to-head and live follow-up tables include High PM2.5
  MAE/Bias/Count.
- A corrected-archive refresh of the test window reproduces previous
  test numbers or documents the revision delta.

## Risks & notes

- Per-line gates increase pipeline branch complexity; keep the
  stage-level structure from 6.1 so data is fetched at most once per
  orchestrator run.
- `trim_prediktioner_log` wipes unrealized predictions at champion
  changes — significance on live data may lose samples; consider
  trimming only at *realized* age (small change to
  `trim_prediktioner_log`, decide explicitly in docs/decisions.md).
- Diebold–Mariano with overlapping h=6 forecasts needs an HAC
  estimator (lag ≥ 5); use the bootstrap as the primary tool for
  readability and testability.
- Significance-gating a champion *switch* is a policy decision
  (threshold, α) — refer it to `config.toml`, and keep the tie rule
  (old champion retained) as the conservative default.
