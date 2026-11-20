# Implementation roadmap: plans 01, 02, 04, 05, 06

Execution plan for implementing the five active limitation-fix plans.
Scope decided 2026-11-15: **plan 07 (scale-out) is implemented by a
separate agent and is out of scope here** — it is neither implemented
nor modified by this series. Plan 03 is an accepted limitation (no
work).

Detailed per-plan designs live in their own files:
[01](01-fix-high-pm-bias.md) · [02](02-champion-composite-selection.md) ·
[04](04-missing-data-bias.md) · [05](05-hyperparameter-optimization.md) ·
[06](06-verification-gaps.md). This roadmap only sequences them.

## Phase order and why

| Phase | Plan | Why this position |
|---|---|---|
| 0 | Baseline & branches | Fix regression reference points before any change |
| 1 | **05 — Hyperparameter optimization** | Top priority, fully standalone; every later comparison benefits from tuned models |
| 2 | **01 — High-PM niche model + routing** | Decided design; its adoption duel must be run *after* 05's tuned models exist |
| 3 | **02 — Composite selection + locked test** | Refines scoring used by 05's searcher and 01's duel; also unblocks 06's test-period policy |
| 4 | **06 — Verification gaps** | Per-line gates multiply with 01's third line; significance tests plug into 01's duel and 02's composite margins |
| 4b (parallel) | **04 — Missing data logging** | Independent of the others; can run alongside any phase after Phase 2 (shares `predict_pipeline.py` with 01) |

---

## Phase 0 — Baseline & branches (½ day)

- [ ] Run the full test suite (`pytest`) on clean main; record pass/fail
      baseline.
- [ ] Run one full pipeline + `live.py --uppföljning` on the current
      champions; snapshot the champion comparison
      (`logs/jamforelse_champion.csv`), champion logs and
      `docs/evaluation.md` 2026-10-23 numbers as the **regression
      reference** (per-plan acceptance tests compare against these).
- [ ] Create one feature branch per plan, named
      `plan/05-hparam`, `plan/01-nisch`, `plan/02-komposit`,
      `plan/06-verifiering`, `plan/04-bortfall`. Only one phase's
      branch is active at a time (see coexistence rules below).
- [ ] Project convention (per team norms): no direct pushes to main,
      PR review by at least one member, never two people in the same
      file on the same day — apply to agents the same way.

## Phase 1 — Plan 05: Hyperparameter optimization (top priority)

Entry: Phase 0 done.
Branch: `plan/05-hparam`.

**Status 2026-11-15: v1 implemented and verified** (48/48 tests, first
real run executed) — remaining work is Phase 1b (see
[05-hyperparameter-optimization.md](05-hyperparameter-optimization.md)
"Phase 1b"); plans 01/05's tuned-vs-default acceptance reports land
with 1b.2–1b.3.

- [x] `[model.hparam_search]` in `config.toml`: per-model grids
      (SVR `C`/`epsilon` first class; Ridge alpha; RF; small grids for
      HGB/LGBM/XGB; Linear Regression excluded).
- [x] `src/train_models.py`: `sok_basta_params()` with
      `TimeSeriesSplit(n_splits=4, gap=horisont)`; scorer parameter —
      default `neg_mean_absolute_error`, optional high-PM-aware scorer
      (MAE + λ·High-PM MAE, λ in config). Use
      `y_train.quantile(0.90)` inside folds, train-period only.
- [x] `build_model_registry(random_state, bast_params)` accepts tuned
      params; `run_model_line()` wires search → registry → tournament.
- [x] Fail-safe: per-family search exceptions → defaults + warning.
- [x] Cache file + winning params into the champion `.json` metadata.
- [x] Tests: leakage-proof folds, scorer fixture, custom-scorer
      fixture, fail-safe path, determinism, defaults locked.
- [x] Docs: mechanism in `docs/evaluation.md`; decision in
      `docs/decisions.md`.

**First-run evidence (2026-11-15):** search chose SVR C=2.0/ε=0.4;
score effect ≈ 0 (vanlig R² 0.6023→0.6017, förskjuten 0.6217→0.6235)
— SVR near a local optimum in the grid. Two defects found:
`stada_models_dir` deletes the cache file, and no duel occurred
(champion logs were missing → automatic candidate win).

### Phase 1b — first-run follow-ups (before Phase 2 is "done done")

Entry: Phase 1 v1 merged.
Branch: `plan/05-hparam` (continuation) — touches the same Phase-1
files, avoids plan 06 territory.

- [ ] **1b.1** Cache path → `logs/` (or `stada_models_dir` exemption)
      so the search record survives cleanup + test.
- [ ] **1b.2** Tuned-vs-default Δ per family from `cv_results_` (the
      default combo is already a grid candidate); `default_score` +
      `forbattring` in info/cache; per-family table in the run log +
      tests.
- [ ] **1b.3** Scorer/grid decision for the next retrain from 1b.2's
      numbers: raise λ / trim SVR ε-grid / defer to plan 02 composite;
      decision in `docs/decisions.md`, winner's tuned-vs-default
      High-PM numbers in `docs/evaluation.md`.
- [ ] **1b.4** (After 1b.1–1b.2) One forced duel run
      (`antal_dagar_sedan = 0` temporarily) so tuned challengers duel
      the tuned 20:38 champions; report the verdict.

**Exit 1b:** cache survives städning; Δ-table in every run;
scorer/grid decision documented; one real duel performed and reported.

## Phase 2 — Plan 01: High-PM niche model with threshold routing

Entry: **Phase 1 merged** (the adoption duel must re-crown against
tuned champions, not default ones).
Branch: `plan/01-nisch`.

**Status 2026-11-15: v1 implemented and verified** (78/78 tests,
incl. 30 new). The first real duel verdict lands at the next retrain
(champion 7-day clocks); forced-duel run follows the plan-05 1b.4
pattern.

- [x] Threshold infrastructure: rule `[model.high_pm]
      threshold_rule = "pm25_p90_training"` (typo "traning" fixed —
      fresh key, no back-compat); computed from training `pm25`
      (value at T — the column the router keys on, *not* the target);
      stored in niche champion `.json` metadata — single source for
      training, router and offline evaluation.
- [x] Filtered tournament: `filtrera_high_pm` + explicit
      `high_pm25_threshold` parameter on
      `train_and_evaluate_models`/`persistence_baseline`; rows with
      `pm25 ≥ threshold`; persistence baseline included (and saved as
      a real champion via `PersistenceModel` when it wins).
- [x] `run_nisch_linje(cfg, …)` in `src/model_pipeline.py`: candidate
      on `Validation High PM2.5 MAE`; **adoption duel vs. the overall
      champion on the same filtered validation rows**; tie or worse →
      no routing (rejected duels are not logged so a stale rejection
      can never block re-evaluation; the duel re-runs every pipeline
      run). Optional old-niche duel: same metric + tie rule.
- [x] Nisch champion persistence: `filprefix="nisch_"`, new log
      `logs/champion_nisch.csv`, `stada_models_dir` feed extended,
      own freshness clock via suffix `" nisch (T+{horisont})"`.
      `trim_prediktioner_log` extended to a *list* of kept models per
      line so ordinary/nisch champion changes never erase each
      other's era.
- [x] `src/orchestrator.py`: nisch line after the vanliga line (own
      freshness clock); routed-system comparison whenever a niche
      champion exists.
- [x] Live router in `src/predict_pipeline.py`: `hamta_nisch_champion`
      (threshold from metadata only) + `bestam_vanlig_champion`
      (boundary `pm25 ≥ threshold`, inclusive, same as the training
      filter); threshold logged as `nisch_threshold` per prediction;
      `prediktioner_live.csv` migrates its header once for the new
      column.
- [x] Routed-system report: `jamfor_routat_system` in
      `src/metrics.py` — ordinary champion vs. routed system vs.
      persistence on validation (total + High-PM + "MAE under gräns"
      proving the router is off below threshold) →
      `logs/jamforelse_champion.csv` (schema-migrated append).
- [x] Tests: threshold p90 source-strictness, filter correctness,
      adoption rule (win/tie/loss/old-niche retention/replacement),
      router boundary (below/at threshold, no-niche), metadata
      round-trip, städning with third feed, trim with multiple kept
      models, log-schema migration, follow-up splits regimes per
      `(linje, modell)`.

**Exit:** duel verdict produced with numbers. If adopted → routing
active live, routed system beats champion on filtered-row High-PM MAE
and total MAE is unchanged below threshold. If rejected → documented
negative result in `docs/evaluation.md`. Tests green; decision entry.

**Remaining for the verdict (team run):** one pipeline retrain (or a
forced duel with `antal_dagar_sedan = 0` per the 1b.4 pattern) to
produce the duel numbers and fill `docs/evaluation.md` + this
roadmap with the outcome.

**Note:** if the verdict is close, that is *exactly* what Phase 4's
significance test is for — tag the duel result and re-adjudicate there.

## Phase 3 — Plan 02: Composite scoring + locked test period

Entry: Phases 1–2 merged (their selection code gets *re-pointed*, not
rewritten, in this phase).
Branch: `plan/02-komposit`.

**Status 2026-11-15: code implemented and verified** (104/104 tests
incl. 30 plan-01 tests). Plan 01 landed first, so its duels were built
on selection-as-reported rather than re-pointed — the niche line keeps
High-PM MAE as its rule by design, composite reported alongside.
**First regression verdict (smoke run on the local Feb–Jul 2026
snapshot, standard-parameter models): persistence-baseline WINS the
composite tournament at all tested weights**
(baseline composite 3,93 vs best model HGB 4,25; baseline nearly
unbiased at peaks while all models underestimate by ≈ −4,5 µg/m³;
see docs/evaluation.md "Kompositens första utvärdering"). Honest
outcome — the formula refuses to crown a peak-blind model — but it
means the next real retrain likely logs "no new champion" while live
keeps serving the existing champion. Remaining team actions:
(1) decide on the verdict — accept, adjust `[model.composite]` weights,
or await plan-05 tuned models (decision → `docs/decisions.md`);
(2) one forced duel (`antal_dagar_sedan = 0`, per the 1b.4 pattern) for
the official numbers;
(3) create the first locked snapshot with `python main.py --ladda-test`.

- [x] `src/metrics.py`: `composite_score(metrics, vikter)` —
      `w₁·MAE + w₂·HighMAE + w₃·|HighBias|`; NaN-safe renormalization
      (threshold not reached / n too small); `[model.composite]`
      weights in config. Add `"Composite": lower-is-better` to
      `metric_direction`.
- [x] Composite column in `train_and_evaluate_models` results;
      candidate selection switches to it
      (
      `candidate_selection_metric = "Validation Composite"`).
- [x] Duel moves to **validation data**: `run_model_line` passes
      `val_path`; `compete_with_champion` decides by composite; tie
      rule unchanged; threshold plumbed into the duel evaluation.
- [x] Locked test snapshot: one-off fetch → `data/test/last/` with
      manifest (source, period, fetch timestamp); a report-only
      evaluation step (never selection) — `jamfor_championer` re-pointed
      at the snapshot, appending to `logs/jamforelse_champion.csv`; 
      refresh policy documented in `docs/decisions.md`.
- [x] Champion log gains composite component columns (`score_mae`,
      `score_high_mae`, `score_high_bias`, schema-migrated); removed
      the hardcoded `"Test R²"`; both `*_selection_metric` keys now
      truthfully read `"Validation Composite"`.
- [x] Re-point Phase 1 and 2 code: the HP-search scorer defers to the
      composite as a parameter drop-in (falls under 1b.3's pending
      scorer/grid decision — untouched in this branch); the niche
      adoption duel stays on High-PM MAE (that is its rule by design)
      but is reported alongside the composite.
- [x] Tests: composite math fixture, NaN fallback, tie/better/worse
      duel, duel-reads-validation not test (path assertion), log
      columns + schema migration, snapshot guard/jämförelse.
- [ ] Weight sensitivity check in a notebook (w₂ ∈ {0.2, 0.35, 0.5})
      before locking defaults (team action).
- [ ] One real duel run + `--ladda-test` snapshot with numbers in
      `docs/evaluation.md` (team action, the acceptance criterion).

**Exit (code):** no automated selection reads test data anywhere
(`run_model_line` has no `test_path` parameter at all; the rolling
fetch is kept only for datakontroll provenance); composite components
logged for all ordinary lines; tests green;
`docs/evaluation.md` "Testdata används vid championvalet" paragraph
replaced by the new procedure. **Exit (team):** the two unchecked
items above.

## Phase 4 — Plan 06: Verification gaps

Entry: Phase 3 merged (gates + significance reference composite
selection; per-line structure includes the nisch line).
Branch: `plan/06-verifiering`.

- [ ] Per-line freshness gates in `src/orchestrator.py`: compute
      freshness for vanlig, forskjuten AND nisch lines; fetch/processing
      runs if *any* line is stale; each stale line retrains; skip
      branch prints all lines' status.
- [ ] Threshold propagation: persist to champion metadata (single
      source of truth). Pass `high_pm25_threshold` to
      `jamfor_championer` (parameter exists, is never passed today).
- [ ] High-PM metrics in `sammanstall_live_uppfoljning` (per unique
      måltid persistence row included).
- [ ] Significance testing in `src/metrics.py`: paired moving-block
      bootstrap (blocks ≥ horizon, seeded, deterministic) as primary;
      Diebold–Mariano with HAC lags ≥ 5 optional secondary. n < 30 →
      "too few observations" guard.
- [ ] Wire in: champion duel (report significance, optionally gate
      saving on p < α when margin tiny — α in config), plan-01 niche
      duel re-adjudication, live follow-up verdict.
- [ ] Provenance: test data marked `latest-months`/preliminary in
      `datakontroll` row + optional corrected-archive refresh job that
      re-reports the locked test window when SMHI publishes it.
- [ ] Revisit `trim_prediktioner_log` (decisions 2026-11-15): decide
      explicitly whether to trim unrealized predictions at challenger
      switch again, since it shrinks significance-test samples.
- [ ] Tests: per-line gate matrix (each stale/fresh combination),
      high-PM propagation, bootstrap determinism + p-value sanity,
      n-guard, provenance flag.

**Exit:** every reported between-model difference carries n, test used,
and p (or an explicit "not yet testable" label); all three lines retrain
on their own 7-day clocks; test-window provenance is explicit.

## Phase 4b (parallel) — Plan 04: Missing data logging

Entry: **Phase 2 merged** (shares `predict_pipeline.py` with 01's
router; otherwise independent).
Branch: `plan/04-bortfall`.

**Status 2026-11-15: implemented and verified** (141/141 tests). The
`find_nan_gaps` helper that predated the plan was extended into
`sammanstall_bortfall` rather than replaced; the day-block pattern is
confirmed with numbers in `docs/data.md` (13 × exactly-24 h
precipitation blocks in the training period; longest PM2.5 gap 93 h —
larger than the 17 h previously assumed).

- [x] Gap/block statistics (`sammanstall_bortfall` →
      `logs/bortfallsanalys_<par>.csv`): per column longest gap, gaps ≥ 6 h
      with start/end/duration/month; run-log summary print.
- [x] `logs/live_skippad_<par>.csv`: one reason row per skip
      (`ingen_ankarpunkt`, `for_gammal_ankarpunkt`,
      `t_obs_redan_predikterad`, `prognos_saknas_maltid`,
      `champion_saknas:<linje>`, `vader_slot_ogiltig:<linje>`,
      `ovantat_fel:<Typ>` for live.py's per-pair catch-all); every vakt in
      `_kor_live_prediktion` instrumented.
- [x] Fix the crash path: missing prognos row for måltid → graceful
      per-line skip (serve the vanliga line; skip the förskjuna) —
      replaces today's `ValueError` that kills both lines. Documented in
      docs/decisions.md.
- [x] Skip summary (7-day counts per reason) printed by
      `live.py --uppföljning` (via `visa_skip_sammanstallning`, shown even
      when nothing has realized yet).
- [x] Tests: gap-stat fixtures (12-h block, 17-h gap), one-row-per-skip
      instrumentation, no-exception guarantee, summary counts.
- [x] `docs/data.md`: batch missing-data pattern summary with numbers
      (+ limitations sentence in `docs/evaluation.md`).

**Exit:** live missing data can never crash a run; every skip is
machine-readable; batch pattern is documented; tests green.

---

## Coexistence with the scale-out agent (plan 07, external)

The other agent owns the station×horizon restructure. To avoid
collisions:

- **Files this series must treat as owned by scale-out (do not
  restructure):** `main.py` loop structure, `src/config.py`'s loading
  machinery (add config *keys* freely — new `[section]`s only — but no
  factory/rename changes), station/path templating semantics in
  `skapa_testvagar`.
- **Hottest shared file:** `src/orchestrator.py` — scale-out wants a
  loop, this series wants the nisch line (Phase 2) and per-line gates
  (Phase 4). Keep our additions small and compositional (a
  `run_nisch_linje` call; a freshness-computation step), which
  multiply cleanly with a future station/horizon loop rather than
  conflict with it. Land Phases 2 and 4 promptly rather than holding
  them back.
- **Config keys added by this series** (register them so the scale-out
  agent can carry them into `for_station_horisont`):
  `[model.hparam_search]`, `[model.high_pm]`, `[model.composite]`,
  `[signifikans]` (α), `nisch_log` path, `logs/bortfallsanalys.csv`,
  `logs/live_skippad.csv`, locked-test manifest path.
- **Per-line pattern compatibility:** everything this series adds is
  "another line" (nisch) or "per-line extension" (freshness, high-PM
  threshold metadata in champion `.json`) — both already namespaced by
  log path/filprefix, which is exactly the mechanism scale-out
  generalizes. No merge surprises expected in `io.py`/
  `model_pipeline.py` if each side keeps to additive functions.
- If the scale-out agent lands first, this series re-branches and
  re-verifies Phase 0 baselines before continuing; the phase order
  among our five plans is unchanged.

## Working rules for every phase

1. One plan = one branch = one PR set; merge a plan fully before
   starting the next (except 4b, which may trail in parallel).
2. Tests: the plan's own test list must pass *plus* the pre-existing
   suite; add regression tests referencing Phase-0 baseline numbers
   where a behavior is supposed to be preserved.
3. Every phase ends with: `pytest` green, one full pipeline run,
   updated `docs/evaluation.md` + new entry in `docs/decisions.md`.
4. Reproducibility: fixed `random_state` everywhere, pinned sklearn
   version, no wall-clock-based sampling.
5. Champions can change as a *result* of a phase (that is the point) —
   but the champion log must always be interpretable: which score
   rule produced the entry, which params, which threshold.
