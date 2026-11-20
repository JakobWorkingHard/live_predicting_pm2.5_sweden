# Plan 01: High-PM2.5 niche model with threshold routing

Part of the limitation-fix series for Predicting Air Quality
(PM2.5, Malmö Dalaplan). Source: report §4–5, tables 1–2; design
decided with the team 2026-11-15. **Priority plan.**

## Limitation

All ML models systematically underestimate high PM2.5 concentrations
(High PM2.5 Bias ≈ −5.8 to −6.1 µg/m³ above the training 90th
percentile ~14.4 µg/m³). At peaks the persistence baseline beats every
ML model, inverting the ranking whenever the purpose is warning for
high levels. Frozen as known limitation on 2026-10-19.

## Decided approach (team feedback 2026-11-15)

Re-run the model tournament **focused only on high PM2.5 values**, and
if the resulting niche model beats the current champion on high-PM
metrics, **route live predictions to it whenever PM2.5 is unusually
high** (above a threshold).

Concretely:

1. **Niche tournament** — train a second tournament of models on
   training rows filtered to high-PM conditions (see threshold below).
   Include the persistence baseline in this tournament (the report
   shows persistence is the best peak performer today — if persistence
   wins the niche tournament, that is a valid and useful outcome).
2. **Niche champion** — selected on high-PM metrics on validation
   (High PM2.5 MAE primary; High Bias as tiebreak/secondary).
3. **Adoption rule** — the niche model is adopted iff it beats the
   *overall champion* on the same high-PM validation rows
   (champion-vs-niche head-to-head on High PM2.5 MAE; tie → no
   routing, ordinary champion serves everything).
4. **Threshold routing** — in live, if the observed `pm25` at T_obs is
   at/above the threshold, the niche champion predicts; otherwise the
   ordinary champion does. Same rule in batch/offline evaluation of
   the routed system.

## Design decisions to lock

- **Threshold**: 90th percentile of *training* `pm25` (~14.4 µg/m³,
  "unusually high"). Recomputed at each retrain, stored in the niche
  champion's `.json` metadata so live and training can never disagree.
  Config key: `[model.high_pm] threshold_rule = "pm25_p90_traning"`.
- **Row filter for niche training**: rows where `pm25` (the value at T,
  i.e. what the router keys on) ≥ threshold. Router, filter and
  evaluation all use the same column and threshold — no mismatch.
- **Niche evaluation set**: validation rows with `pm25` ≥ threshold —
  exactly the rows where routing would serve the niche model. This is
  the fairest comparison of "champion vs. routed system".

## Current state (verified 2026-11-15)

- `src/metrics.py::evaluate_model()` / `calculate_high_pm25_metrics()`
  already compute High PM2.5 MAE/Bias/Count given a threshold —
  reusable as-is.
- `src/train_models.py::train_and_evaluate_models()` runs the
  tournament but on the full dataset only; no filtered tournament.
- Champion machinery is per-line already: separate logs
  (`champion_vanlig.csv`, `champion_forskjuten.csv`), name-suffix-locked
  freshness (`champion_is_fresh(namn_suffix=…)`), `load_latest_champion`
  via the `sokvag` log column, `stada_models_dir` per line — a third
  "nisch" line plugs into this pattern cleanly.
- Live: `src/predict_pipeline.py::kor_live_prediktion()` knows
  `pm25_varde` at T_obs before predicting — the router hook is
  trivially placed; `sammanstall_live_uppfoljning()` groups by
  `(linje, modell)`, so routed predictions are automatically reported
  as their own row in the follow-up.

## Implementation steps

1. `src/train_models.py` — add `train_and_evaluate_models_high_pm(...)`
   (or a `filter_high_pm=True` path) that:
   - computes the threshold from training `pm25` (p90), prints it;
   - filters X_train/y_train and X_val/y_val to `pm25 ≥ threshold`
     rows;
   - runs the same registry + persistence baseline on the filtered
     data, reusing `evaluate_model` with threshold.
2. Niche selection & duel — in `src/model_pipeline.py` (new
   `run_nisch_linje(cfg, …)`, mirrors `run_model_line`):
   - select niche candidate on `"Validation High PM2.5 MAE"`;
   - load the *overall* champion and evaluate it on the same filtered
     validation rows; adopt the niche champion only if it wins on High
     PM2.5 MAE (tie → keep no-routing);
   - save as `save_champion_model(..., filprefix="nisch_")`,
     log to `logs/champion_nisch.csv` (new path in config), metadata
     includes `threshold` value and rule.
   - If a niche champion already exists: optional duel of
     niche-vs-niche (new candidate vs. old niche champion), same
     high-PM metric and tie rule as the ordinary lines.
3. Orchestrator — after the vanliga line: run the nisch line
   (checks its own `champion_is_fresh` with suffix
   `" nisch (T+{horisont})"`); extend `stada_models_dir` feed list
   with `(cfg.nisch_log, "nisch_")`.
4. Live routing — in `kor_live_prediktion()`:
   - after building the vanlig line's row, read the active regime from
     the niche champion metadata (only if a niche champion exists);
   - if `pm25 at T_obs ≥ threshold` → predict with the niche champion
     for the vanliga line (mätlidsväder-loop can adopt the same router
     later, its own niche model as a follow-up);
   - the logged `modell` name already carries "nisch", so uppföljning
     splits regimes automatically; also log the threshold used.
5. Offline evaluation of the routed system — extend
   `jamfor_championer()` (which already accepts a threshold) or add a
   small report comparing: ordinary champion vs. routed system
   (champion + niche) on validation/test, on both total and High-PM
   metrics. Result table to `logs/jamforelse_champion.csv`.
6. Docs — decision entry (unfreezing 2026-10-19 with the niche
   approach), results in `docs/evaluation.md`, README caveat about
   high-level warnings updated per outcome.

## Affected files

- `src/train_models.py` — filtered tournament
- `src/model_pipeline.py` — `run_nisch_linje`, adoption duel
- `src/champion.py` — reuse as-is (log-driven loading)
- `src/io.py` — champion metadata gains `threshold`; `stada_models_dir`
  new feed
- `src/orchestrator.py` — third line
- `src/predict_pipeline.py` — router + log fields
- `src/metrics.py` — routed-system comparison helper (optional)
- `config.toml` / `src/config.py` — `nisch_log` path, threshold rule
- `tests/test_train_models.py`, `tests/test_champion.py`,
  `tests/test_predict_pipeline.py`, `tests/test_io.py`

## Tests

- Threshold: p90 computed from training `pm25` only, never from
  validation/test; stored to and read from metadata identically.
- Filter: niche tournament sees only rows with `pm25 ≥ threshold`;
  evaluation rows likewise.
- Adoption rule: niche wins → saved + routed; tie or worse → no routing
  (router reads "no niche champion" state); old champion retains when
  overall-champion wins.
- Router: `pm25` one µg/m³ below threshold → ordinary champion;
  at/above → niche champion; both rows contain exactly `cfg.features`.
- Summary: follow-up groups by modell name → separate rows for niche
  predictions with correct count.

## Acceptance criteria

- A niche champion is trained, champion-logged, and either adopted
  (routing live) or rejected with a documented, reproducible negative
  result in `docs/evaluation.md`.
- If adopted: on the filtered validation rows, routed-system High PM2.5
  MAE < ordinary-champion High PM2.5 MAE, and total MAE outside the
  filtered rows is unchanged (router provably off below threshold).
- High-PM metrics of ordinary champion vs. routed system visible in
  the run output and in `logs/jamforelse_champion.csv`.

## Risks & notes

- **Sample size**: p90 of ~21.5k training rows ≈ 2.1k rows — enough for
  linear/SVR models, tight for tuned boosting ensembles; prefer
  simpler niche models and report episode count; avoid hyperparameter
  search on this subset unless plan 05 handles the budget.
- **Seasonal drift**: p90 of the training window moves between retrain
  runs — that's fine (stored per champion era), but comparisons across
  eras in the uppföljning should use the era's threshold (already
  grouped per modell name per era via `trim_prediktioner_log`).
- **Persistence may win the niche tournament** — then routing simply
  means "at high levels, output the current measurement", which is a
  defensible, explainable system and should be presented as a success
  of the evaluation framework, not a failure.
- Interacts with plan 05: tuned ordinary models (smaller SVR ε, etc.)
  may shrink the niche advantage — re-run the adoption duel after
  plan 05 lands.
