# Plan 05: Systematic hyperparameter optimization — TOP PRIORITY

Part of the limitation-fix series for Predicting Air Quality
(PM2.5, Malmö Dalaplan). Source: report §3 (untried item #1) and
self-reflection; prioritized as **the most important plan** by team
decision 2026-11-15.

## Limitation

All seven competing models are trained with library-default
hyperparameters. From `src/train_models.py::build_model_registry()`:

| Model | Fixed defaults |
|---|---|
| SVR | `kernel="rbf", C=1.0, epsilon=0.1` |
| Ridge | `alpha=1.0` |
| Random Forest | `n_estimators=200` |
| HistGradientBoosting | all defaults |
| LightGBM | all defaults (`verbose=-1`) |
| XGBoost | all defaults (`verbosity=0`) |
| Linear Regression | n/a |

The report explicitly lists systematic optimization of SVR's `C` and
`epsilon` as a top untried idea. Tuning ε is doubly interesting: it
controls the ε-insensitive band and therefore how much the model
smooths towards "dämpad persistence" — directly related to the
high-PM underestimation (plan 01).

## Goal

A reproducible, leakage-free hyperparameter search integrated into the
existing tournament: every model still competes, but on tuned
settings, with chosen parameters recorded, reusable live, and —
critically — **runnable now**, without waiting for any other plan.

## Implementation steps

1. **Search grids in config** — `[model.hparam_search]`:
   - SVR (highest priority per the report):
     `C ∈ {0.5, 1, 2, 4, 8}`, `epsilon ∈ {0.05, 0.1, 0.2, 0.4}` — it
     is *specifically* interesting whether a smaller ε improves the
     high-PM bias.
   - Ridge: `alpha ∈ {0.1, 0.3, 1, 3, 10}`
   - RF: `n_estimators ∈ {100, 200, 400}`, `max_depth ∈ {None, 10, 20}`
   - HGB / LightGBM / XGBoost: small deliberate grids
     (learning_rate, n_estimators/max_iter, num_leaves/max_depth) with
     a fit budget (see notes). Linear Regression: skipped.
2. **Search harness in `src/train_models.py`** —
   `sok_basta_params(modell_namn, X_train, y_train, cfg)`:
   - `sklearn.model_selection.*SearchCV` with
     `cv=TimeSeriesSplit(n_splits=4, gap=horisont)` — splits **inside
     the training period only**; validation stays untouched, test
     remains unseen for selection (unchanged policy).
   - **Scoring**: `neg_mean_absolute_error` as the default interim
     scorer; additionally support a high-PM-aware custom scorer
     (MAE + λ·High-PM MAE on the fold, λ configurable, e.g. 0.5 —
     small standalone helper, no dependency on plan 02). Log the
     High-PM metrics of the winning configurations either way, so the
     tuned-vs-default comparison at the peaks is always visible.
   - Deterministic: fixed `random_state`, no wall-clock subsampling.
   - Cache results to `models/hparam_sok_<datum>.json` so every
     champion change is explainable.
3. **Registry update** — `build_model_registry(random_state, bast_params)`
   instantiates tuned estimators when search results exist;
   `run_model_line()` wiring: search (once per retrain run, shared per
   model family) → registry → tournament as today. Fail-safe: search
   error → defaults + loud warning.
4. **Metadata & docs** — winning params into the champion `.json`;
   top-3 configs per model into docs/evaluation.md; high-PM metrics
   for tuned vs. default SVR reported; decision entry in
   docs/decisions.md.
5. **Follow-up hook for plan 01** — after tuning lands, re-run the
   niche-model adoption duel: tuned ordinary models may shrink (or
   widen) the niche advantage; the duel decides with data.

## Affected files

- `src/train_models.py` — grids (from config), search harness,
  high-PM-aware scorer, tuned registry
- `src/model_pipeline.py` — wire search into `run_model_line`
- `config.toml` / `src/config.py` — grids, λ, budget knobs
- `src/io.py` — champion metadata params, search-result cache file
- `tests/test_train_models.py`

## Tests

- `TimeSeriesSplit` folds are chronological with `gap=horisont`; a
  fold's train window never overlaps the evaluation window's target
  horizon (leakage unit test on a synthetic index).
- One-value grids reproduce today's default registry exactly.
- Search is deterministic given `random_state` and a fixed sklearn
  version (pin in `requirements.txt`).
- Custom scorer: hand-computed fixture — high-PM-aware score equals
  MAE + λ·High-MAE on rows above the fold threshold.
- Fail-safe: a model family whose search raises trains with defaults
  and logs a warning.

## Acceptance criteria

- Champions are produced by tuned configurations; the chosen params
  are visible in `.json` metadata and the champion log.
- For every model family: tuned vs. default comparison reported on
  **MAE and High-PM MAE/Bias** (validated on the validation set — no
  silent regressions anywhere).
- Full tournament runtime within the agreed budget (target: today's
  runtime × < 5; measured and documented).
- SVR specifically: best (`C`, `ε`) documented, with the effect of
  smaller ε on the peak bias stated with numbers.

## Risks & notes

- **Runtime**: grids × 4 folds × ~21.5k rows is cheap for
  linear/Ridge, moderate for SVR (O(n²)–O(n³) — subsample the search
  if needed, note it in the cache), expensive for exhaustive boosting
  grids — prefer small grids or `HalvingRandomSearchCV`; the budget
  knob lives in config.
- **Interaction with plan 01**: smaller ε and tuned trees may
  themselves reduce peak underestimation; hence "re-run the niche
  duel" in step 5 — the routing threshold might then key on a
  slightly different model ranking.
- **Interaction with plan 02** (whenever it lands): switch the search
  scorer to the full composite formula at that point; the harness
  takes the scorer as a parameter, so this is drop-in.
- Keep searches reproducible and version-pinned; run inside the
  normal 7-day champion freshness cadence (search piggybacks on
  retrain runs, never on live runs).

---

## Status 2026-11-15 — v1 implemented; Phase 1b follow-ups (from the first real run)

### What landed (Phase 1 v1)

- `src/train_models.py`: `high_pm_score` / `high_pm_scorer`
  (sklearn ≥1.6 scorer convention `(estimator, X, y)` — the legacy
  `(y_true, y_pred)` signature no longer works and was caught by the
  tests), `sok_basta_params` (TimeSeriesSplit gap=horisont, refit=False,
  tail-subsample for SVR), `sok_parametrar_for_registry` (fail-safe,
  in-process memo shared between model lines), `spara_hparam_cache`,
  `build_model_registry(random_state, bast_params=None)` with today's
  defaults locked by regression test.
- `config.toml` / `src/config.py`: `[model.hparam_search]` + per-family
  grids; `src/model_pipeline.py` wires search → registry → tournament;
  champion metadata carries `hparam_sok` (`src/io.py`).
- +10 tests (48 total, 38 baseline green). Docs: `docs/decisions.md`,
  `docs/evaluation.md` (mechanism section).

### First real run (2026-11-15 ~20:38) — evidence table

| | 19:54 run (defaults) | 20:38 run (tuned) |
|---|---|---|
| SVR params | C=1.0, ε=0.1 | **C=2.0, ε=0.4** (both lines) |
| Test R² vanlig | 0.6023 | 0.6017 |
| Test R² förskjuten | 0.6217 | 0.6235 |

Findings:

1. The search works end-to-end (non-default params chosen, shared
   between lines, recorded in champion metadata) but its effect on
   scores was ≈ 0: SVR sits near a local optimum within this grid.
2. **BUG — cache destroyed by cleanup:** `stada_models_dir` deleted
   `models/hparam_sok_2026-11-15_203755.json` immediately after the run
   ("Städade models: tog bort 1 modellfil(er): hparam_sok…"). The full
   per-family search record (all families, fold scores, chosen params)
   is lost every run; only the winning family's params survive in the
   champion metadata.
3. The champion logs were missing at 20:38, so there was **no duel** —
   candidates won automatically (fail-safe design). The
   tuned-vs-previous-champion duel the DoD requires has therefore not
   actually happened yet.
4. **Scorer warning:** with λ=0.5 the search still chose a *wider*
   ε (0.4 = more smoothing, likely worse at peaks). The high-PM
   signal in the CV folds is too weak at λ=0.5 to push ε down.

### Phase 1b tasks (remaining Phase-1 DoD)

1. **Cache must survive cleanup (bug fix).** Recommended: write the
   cache to `logs/hparam_sok_<timestamp>.json` instead of `models/` —
   `stada_models_dir` never touches logs/, and `models/` stays a pure
   champion store. (Alternative: exempt `hparam_sok_*` inside
   `stada_models_dir`.) Update `spara_hparam_cache` + the config/test
   accordingly. Test: a `stada_models_dir` run leaves the cache file
   in place.
2. **Tuned-vs-default visible in every run** — zero-extra-fits design:
   the default parameter combination is already a candidate in every
   grid, so read `gs.cv_results_`, locate the default candidate's
   `mean_test_score`, and add to each family's info dict:
   `default_score`, `forbattring = default_score - best_score`
   (≥ 0, since scores are negative errors). Persist both in the cache
   JSON and print a compact per-family table (chosen params, search
   score, default score, Δ) to the run log. Tests: Δ ≈ 0 when the
   defaults win the grid; Δ > 0 for a known-better candidate.
3. **Scorer/grid review at the next retrain** — decide from 1b.2's
   table: (a) raise λ (e.g. 1.0–2.0), (b) drop ε ∈ {0.2, 0.4} from the
   SVR grid if they win on total MAE while losing on High-PM, or
   (c) defer to plan 02 and plug the composite in as the search
   scorer (drop-in through `high_pm_scorer`). Log the decision in
   `docs/decisions.md`; report tuned-vs-default High-PM numbers for
   the winner in `docs/evaluation.md` (this also closes the original
   "tuned vs. default SVR at peaks" acceptance criterion).
4. **(Operational) One forced duel.** After 1b.1–1b.2 land: run once
   with `antal_dagar_sedan = 0` temporarily so tuned challengers
   actually duel the 20:38 tuned champions — the comparison missing
   from the first run. The tie rule protects the champions (switch
   only on strictly better score). Do this AFTER 1b.1, or the new
   cache evidence gets deleted too.

**Exit criteria 1b:** cache survives städning; tuned-vs-default table
in every run's log + cache; scorer/grid decision documented (or
explicitly deferred to plan 02 with motivation); one real duel
performed and reported.
