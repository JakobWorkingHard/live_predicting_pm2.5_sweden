# Plan 02: Weighted composite formula in champion selection + locked test data

Part of the limitation-fix series for Predicting Air Quality
(PM2.5, Malmö Dalaplan). Source: report §5 ("Fördjupad granskning:
Championvalets urvalsformel och testdataanvändning"); verified against
the codebase 2026-11-15.

## Limitation

The champion choice is driven by single, global mean metrics that are
blind to the −6 µg/m³ peak bias, and the duel reuses rolling test data:

1. Candidate selection: lowest **Validation MAE** alone
   (`config.toml: candidate_selection_metric = "Validation MAE"`).
2. Champion duel: a single **R²** decides replacement — and although the
   config key reads `champion_selection_metric = "Validation R²"`, the
   string is only used to *extract the metric key*; the actual duel in
   `compete_with_champion()` reads **test data** (`test_path =
   testvagar["test_processed"]`, fetched fresh from rolling
   `latest-months` every run). The label is therefore a misnomer.
3. Repeated duells against a rolling test window risk adaptive
   overfitting to the test period; R² also swings with the season of
   the test window (report; docs/evaluation.md "Testdata används vid
   championvalet").

The frozen-test decision from 2026-10-15 was superseded by the
dynamic-periods decision 2026-10-20, so today every duel touches fresh
test data.

## Current state (verified 2026-11-15)

- `src/champion.py::compete_with_champion()` —
  `test_df = pd.read_csv(test_path)`; both models evaluated with
  `evaluate_model()` **without** a High-PM threshold; winner by
  `_metric_key_and_direction(metric)` on a single key.
- `src/model_pipeline.py::run_model_line()` — candidate from
  `select_best_model(comparison_df, metric=cfg.candidate_selection_metric)`;
  duel with `test_path=testvagar["test_processed"]`;
  `champion_test_score = test_results_df.iloc[winner_index]["R2"]`;
  champion log entry `metric="Test R²"` hardcoded.
- `src/metrics.py` — `select_best_model`, `metric_direction`,
  `evaluate_model` (already supports High-PM metrics via threshold),
  `build_comparison_table`.
- `src/orchestrator.py` — dynamic test via `fetch_and_merge_test(cfg)`;
  the whole pipeline is skipped when the vanliga champion is fresh.
- Champion logs: `logs/champion_vanlig.csv`,
  `logs/champion_forskjuten.csv` (columns include `metric`, `score`,
  `sokvag`).

## Goal

1. Replace the single-metric candidate choice and duel with the report's
   weighted composite formula:
   **Total score = w₁·MAE(total) + w₂·MAE(high) + w₃·|Bias(high)|**
   (lower is better; NaN-safe fallback).
2. Run the champion duel on **validation data** so a separate, locked
   test period can be kept for independent evaluation (the report's
   explicit recommendation), eliminating adaptive overfitting on test.
3. Fix the misleading `champion_selection_metric` naming.

## Implementation steps

1. **Composite score in `src/metrics.py`** — new
   `composite_score(metrics_dict, vikter) -> float`:
   - Components: `MAE`, `High PM2.5 MAE`, `|High PM2.5 Bias|`.
   - Weights in `config.toml` (e.g. `[model.composite] w_mae = 0.5,
     w_high_mae = 0.35, w_high_bias = 0.15`).
   - NaN handling: if `High PM2.5` metrics are NaN (no peaks in the
     slice) or count is tiny (< e.g. 10 rows), renormalize to the
     remaining weights and log a warning.
   - `evaluate_model()` is already threshold-aware; make
     `run_model_line` compute the threshold once from `y_train` and
     pass it consistently (also to `compete_with_champion`).
2. **Composite-based candidate selection** — in
   `train_and_evaluate_models()` add a `"Composite"` column computed per
   model; `select_best_model()` sorts by it
   (`cfg.candidate_selection_metric = "Validation Composite"`).
   `metric_direction()` gains a `"Composite": "lower"` branch.
3. **Move the duel to validation data** —
   `run_model_line()` passes `val_path` instead of `test_path`;
   `compete_with_champion()` (rename param to `data_path`) evaluates both
   models there with threshold, decides by composite score; tie rule
   unchanged (old champion retained — keep!).
4. **Locked test period** — reinstate the 2026-10-15 idea, adapted to the
   dynamic setup:
   - One-off snapshot job (flag to `main.py`, e.g. `--ladda-test`) that
     fetches the test window from `latest-months` and freezes it under
     `data/test/last/` with a manifest noting PM2.5 series id, weather
     station id, and fetch timestamp.
   - An independent evaluation step (report-only, not selection) that
     scores all champions on the locked snapshot; results appended to
     `logs/jamforelse_champion.csv` or a new
     `logs/sluttest_<datum>.csv`.
   - A documented refresh policy: re-snapshot at most per major version,
     always noted in `docs/decisions.md`.
5. **Champion log fix** — log the composite components
   (`metric="Composite", score=…, score_mae=…, score_high_mae=…,
   score_high_bias=…`) for both model lines; remove the hardcoded
   `"Test R²"`.
6. **Config naming cleanup** — replace misleading keys:
   - `candidate_selection_metric = "Validation Composite"`
   - `champion_selection_metric = "Validation Composite"` (now truthful:
     the duel really runs on validation data)
   - or introduce `[model.composite]` and drop the string-metric keys
     entirely.
7. **Update docs** — `docs/evaluation.md` sections "Vad vi mäter",
   "Datauppdelning", "Begränsningar" (remove/replace the
   "Testdata används vid championvalet" paragraph); decision entry in
   `docs/decisions.md`.

## Affected files

- `src/metrics.py` — composite score, direction handling
- `src/champion.py` — duel on validation, param renames
- `src/model_pipeline.py` — wiring, threshold plumbing, log fields
- `src/train_models.py` — Composite column in results_df
- `src/config.py` / `config.toml` — weights, metric keys
- `src/orchestrator.py` — pass val paths / test snapshot paths correctly
- `src/io.py` — champion-log writer fields
- `tests/test_metrics.py`, `tests/test_champion.py`,
  `tests/test_train_models.py`

## Tests

- `composite_score`: expected value on hand-computed fixture; NaN
  high-metrics → renormalized weights; zero count → warning + fallback.
- Duel: new model wins only if *strictly* better composite score; tie or
  worse → old champion kept; missing old champion → new wins (unchanged).
- Duel reads the validation file, never the test file (assert on the
  path argument).
- Champion log contains composite component columns.
- Regression: full `run_model_line` run on the 2026-10-23 data snapshot
  keeps selecting an SVR-family champion (sanity check that scoring
  change alone doesn't crown persistence).

## Acceptance criteria

- No automated selection decision reads test data anymore; a locked test
  snapshot exists and is used only for reporting.
- Composite formula with configurable weights drives both candidate
  choice and champion duel, and components are logged.
- The config keys' names match the data actually used.
- Champion/Challenger pattern and tie rule preserved (decisions log
  2026-10-03 heritage).

## Risks & notes

- Weight choice is a judgment call; make weights config-driven, test
  sensitivity (e.g. w₂ ∈ {0.2, 0.35, 0.5}) in a notebook before locking.
  This choice is itself the report's core lesson: "systemet blir aldrig
  bättre än den matematiska utvärderingsformel man har kodat in".
- Moving the duel to validation means validation partly loses
  "untouched" status; acceptable per the report (validation is made
  for selection), and the locked test restores independent evaluation.
- Beware interactions: plan 05 (hyperparameter search) must select on
  training-internal folds, not on the validation used here.
