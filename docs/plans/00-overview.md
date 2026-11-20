# Overview: Fixing the limitations of Predicting Air Quality

Series of plans addressing the limitations identified in the individual
analysis report (Jakob Ohlsén S., 2026-10-30), re-verified against the
codebase on 2026-11-15 and reprioritized by team feedback the same day.

## Plans — team-prioritized order (2026-11-15)

| Order | File | Limitation | Status |
|---|---|---|---|
| 1 | [05-hyperparameter-optimization.md](05-hyperparameter-optimization.md) | All 7 models on default hyperparameters (SVR `C=1.0, ε=0.1`…) | **v1 implemented 2026-11-15** (48/48 tests; first run picked C=2.0/ε=0.4, effect ≈ 0). **Phase 1b pending:** cache-survival fix, tuned-vs-default Δ table, scorer/grid decision, one forced duel |
| 2 | [01-fix-high-pm-bias.md](01-fix-high-pm-bias.md) | High-PM bias ≈ −6 µg/m³ → niche model trained on high-PM rows only, adopted if it beats the champion on high-PM metrics, routed live above a threshold | **Priority** — decided design |
| 3 | [02-champion-composite-selection.md](02-champion-composite-selection.md) | Single-metric champion selection on reused/rolling test data (`"Validation R²"` key is a misnomer — the duel still runs on test data) | **Code implemented 2026-11-15** (104/104 tests; composite selection + validation duel + locked-test snapshot `--ladda-test`). First verdict: **persistence wins the composite on the current snapshot at all tested weights** — team decision pending (accept / adjust weights / await tuned models) |
| 4 | [06-verification-gaps.md](06-verification-gaps.md) | Per-line freshness gates, High-PM on test/live, significance tests, preliminary-data flagging | Supporting — closes the evidence loop |
| 5 | [04-missing-data-bias.md](04-missing-data-bias.md) | Keep dropna policy + proper logging; **live-mode missing data** made explicit, graceful and logged (today: silent skips, one crash path) | **Implemented 2026-11-15** (141/141 tests; per-pair skip log + gap-pattern logging + crash fix; day-blocks confirmed with numbers in `docs/data.md`) |
| — | [07-scale-out.md](07-scale-out.md) | One station, one horizon → all Swedish stations × horizons 1–24 h | **Owned by a separate agent — not part of this series** (file kept as context/requirements) |
| — | [03-train-serve-skew-t6.md](03-train-serve-skew-t6.md) | T+6 line trained on observed ("facit") weather, served forecasts | **ACCEPTED limitation** — dropped, file kept as decision record |

Execution sequencing, branches, phase entry/exit criteria and
coexistence rules with the scale-out agent are in
[implementation-roadmap.md](implementation-roadmap.md).

## Team decisions recorded 2026-11-15

1. **Train–serve skew (former plan 03): accepted.** The oracle-based
   T+6 evaluation is intentional (the line answers a weather-conditional
   question); drift performance stays observable via
   `live.py --uppföljning`. Hindcast + Δ-weather work dropped; revisit
   only if the förskjuna line becomes the recommended production model.
2. **Missing data batch handling: keep `dropna`, log the pattern.** No
   interpolation, no fill+indicator features. New scope: gap/block
   statistics in the batch logs, and — the real open question —
   explicit graceful handling of missing data **at live-prediction
   time** (skip log + no-crash behavior), because today a missing
   forecast row for the måltid crashes the live run, and over-age
   anchor hours are silently skipped with no trace.
3. **Hyperparameter optimization (05): most important plan.** Designed
   to run standalone (MAE scorer by default, optional high-PM-aware
   scorer), with a follow-up hook: re-run plan 01's niche adoption duel
   after tuning lands.
4. **High-PM approach (01) decided:** re-run the tournament focused
   only on high PM2.5 rows; if the niche model beats the champion on
   high-PM metrics, route live predictions to it when PM2.5 exceeds a
   threshold (default: training-set p90 ≈ 14.4 µg/m³, stored per
   champion era).

## Original limitation verification (2026-11-15, still valid)

| Report's limitation | Codebase finding |
|---|---|
| High-PM bias −6 µg/m³ | Unfixed; default losses; High-PM metrics exist in `metrics.py` but never drive selection; `jamfor_championer()` has an unused `high_pm25_threshold` param |
| Single-metric selection + test reuse | Unfixed; `compete_with_champion()` reads `test_path` (rolling `latest-months`) despite the `"Validation R²"` config key |
| Train–serve skew | Real but accepted (see decisions) |
| Non-random missing data | **Fixed 2026-11-15 (plan 04):** pattern logged per pair (`bortfallsanalys` + datakontroll gap columns), day-blocks confirmed with numbers in `docs/data.md`; live-mode missing data explicit, graceful and logged (per-pair `live_skippad` log), crash path removed |
| No hyperparameter tuning | Unfixed; `build_model_registry()` all defaults |
| Verification gaps | Unfixed; freshness gate only the vanliga champion (deliberate); no significance tests; test data preliminary `latest-months` |
| Sketch: one station / one horizon | Architecture supports scale-out (config.toml + orchestrator), not implemented |

## Constraints valid for all plans (from report + decisions log)

- No interpolation of missing values (decisions 2026-10-09/12);
  dropna policy confirmed by team 2026-11-15.
- Champion/Challenger pattern and the tie rule (old champion retained)
  are kept; only metrics/evaluation change (plan 02).
- Data periods are dynamic: training from 2023-05-26, validation = last
  6 whole months of `corrected-archive`, test = rolling `latest-months`
  (decisions 2026-10-20/21).
- Preservation of 2026-11-15 hygiene work: `stada_models_dir`,
  `trim_prediktioner_log` (per champion era), `stada_prognos_dir`
  (30-day retention), log rotation — new plans must plug into these
  (e.g. plan 01 extends `stada_models_dir` feeds with the nisch line).
- SMHI always credited (CC BY 4.0).
- The result must not be presented as suitable for health decisions or
  high-level warnings until plans 01/06 measurably improve peak
  performance.
