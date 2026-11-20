# Updated plan: remaining work in the plans series

Audit date: 2026-11-20. This file supersedes the sequencing in
[implementation-roadmap.md](implementation-roadmap.md) for everything still
open. The per-plan designs remain the source of truth:
[01](01-fix-high-pm-bias.md) · [02](02-champion-composite-selection.md) ·
[05](05-hyperparameter-optimization.md) · [06](06-verification-gaps.md).

## Verified current state (2026-11-20 audit)

- Test suite: **141/141 green** (`.venv\Scripts\python.exe -m pytest`).
- Plan 05 v1 (hparam search), plan 01 (niche line + routing), plan 02
  (composite selection + locked test), plan 04 (missing-data logging):
  **code landed and verified** — confirmed in `src/` and `config.toml`.
- Plan 03: accepted limitation, no work (see
  [00-overview.md](00-overview.md) decisions). Plan 07: separate agent,
  untouched by this series.
- ⚠️ **Workspace is not a git repo** — the original roadmap's branch-per-phase
  workflow (`plan/05-hparam`, …) cannot run as written. Either `git init` +
  commit the current green state first (recommended; enables the team's
  no-direct-push rule and per-phase PRs), or work sequentially on one copy.

## Remaining work, grouped

| Group | Items | Type |
|---|---|---|
| A. Plan 05 Phase 1b | 1b.1 cache-survival fix, 1b.2 tuned-vs-default Δ table | Code |
| B. Verdict run | One forced-duel run + `--ladda-test` snapshot; produces 05's 1b.4 duel, 01's adoption verdict, 02's official composite numbers | Operations |
| C. Team decisions | 1b.3 scorer/grid; 02 weight sensitivity + accept/adjust; 01 adopt/reject (tag for re-adjudication if close) | Decisions |
| D. Plan 06 | Per-line freshness gates, High-PM everywhere, significance testing, provenance flagging | Code |
| E. Docs & closure | evaluation/decisions/data/roadmap/overview updates; final full run | Docs |

Key ordering constraint: **A must land before the forced run (B)**, or the
run's cache evidence is deleted by `stada_models_dir` again. All decisions
(C) consume B's numbers, so B is a single gate for the whole series.

---

## Stage A — Plan 05 Phase 1b (code)

Files: `src/train_models.py`, `src/config.py`, `config.toml`, `src/io.py`,
`tests/test_train_models.py` — no plan-06 territory.

- [ ] **A.1 (= 1b.1)** Move the search cache from `models/hparam_sok_*.json`
      to `logs/hparam_sok_<timestamp>.json` (`spara_hparam_cache` + config
      key). `models/` stays a pure champion store; `stada_models_dir` never
      touches `logs/`.
      Test: a full pipeline run incl. `stada_models_dir` leaves the cache file.
- [ ] **A.2 (= 1b.2)** Zero-extra-fits Δ table: the default combo is already
      a grid candidate — read `gs.cv_results_`, add `default_score` +
      `forbattring` per family to the info dict, persist both in the cache
      JSON, print a compact per-family table (chosen params, search score,
      default score, Δ) in the run log.
      Tests: Δ ≈ 0 when defaults win; Δ > 0 for a known-better candidate.

**Exit A:** cache survives städning; Δ table in every run's log + cache.

## Stage B — One forced verdict run (operations, after A)

Single run with `antal_dagar_sedan = 0` (and the förskjuten clock) so tuned
challengers actually duel the tuned 20:38-era champions. This one run closes
three pending verdicts simultaneously:

- [ ] **B.1 (= 1b.4)** Plan 05 duel: tuned challengers vs tuned champions —
      strictly-better rule protects champions; a tie verdict is a valid
      result.
- [ ] **B.2** Plan 01 adoption duel verdict: niche champion vs overall
      champion on High-PM-filtered validation rows → adopt or reject.
- [ ] **B.3** Plan 02 official composite numbers at current weights
      (the smoke run's persistence-wins verdict re-confirmed on real
      champions).
- [ ] **B.4** Plan 02 locked test snapshot: `python main.py --ladda-test`
      → snapshot + manifest in `data/test/last/`, then one pipeline run so
      `jamfor_championer` reports against the locked snapshot
      (report-only; never selection).
- [ ] Record all numbers in `docs/evaluation.md`; verdicts in
      `docs/decisions.md`; tick the roadmap/overview checkboxes.

**Exit B:** all three verdicts exist as numbers in the docs, not just smoke
evidence.

## Stage C — Team decision gate (uses B's numbers)

- [ ] **C.1 (= 1b.3)** Scorer/grid decision: (a) raise λ (1.0–2.0),
      (b) trim SVR ε ∈ {0.2, 0.4} from the grid if they win on total MAE
      while losing at peaks, or (c) plug in `composite_score`/`high_pm_scorer`
      as the search scorer (drop-in, defers to plan 02). Decision →
      `docs/decisions.md`; winner's tuned-vs-default High-PM numbers →
      `docs/evaluation.md`.
- [ ] **C.2** Plan 02 composite weights: run the weight-sensitivity notebook
      (w₂ ∈ {0.2, 0.35, 0.5}) on B.3's outputs; accept the persistence-wins
      verdict, adjust weights, or await tuned models. Decision →
      `docs/decisions.md`.
- [ ] **C.3** Plan 01 routing: adopt / reject B.2's verdict. If the margin is
      close, **tag it for Stage D's significance re-adjudication** instead of
      deciding on point estimates.

## Stage D — Plan 06 implementation (code, after C)

Design: [06-verification-gaps.md](06-verification-gaps.md) §6.1–6.4.
Current state re-confirmed by the 2026-11-20 audit: single-line gate in
`orchestrator.py`; `high_pm25_threshold` param exists but is never passed to
`jamfor_championer`; no significance machinery anywhere in `src/`.

- [ ] **D.1 (§6.1)** Per-line freshness gates in `src/orchestrator.py`:
      compute freshness for vanlig, förskjuten AND nisch; fetch/processing
      runs if any line is stale; each stale line retrains; skip branch prints
      all lines' status. Keep the champion-name suffix horizon protection.
      Keep additions compositional (per the plan-07 coexistence rules —
      no loop restructuring in orchestrator/main).
- [ ] **D.2 (§6.2)** Threshold propagation: champion `.json` metadata as
      single source; pass `high_pm25_threshold` to `jamfor_championer`
      (already exists, never passed today); High-PM MAE/Bias columns in
      `sammanstall_live_uppfoljning` (per unique måltid persistence row,
      same dedup rules); one-off High-PM split of the final champions on the
      locked test snapshot.
- [ ] **D.3 (§6.3)** Significance testing in `src/metrics.py`:
      paired moving-block bootstrap (blocks ≥ horizon, seeded, deterministic)
      primary; Diebold–Mariano w/ HAC lags ≥ 5 optional secondary; n < 30 →
      "too few observations" guard. Wire into the champion duel (report p;
      optionally gate saving on p < α when the composite margin is tiny,
      α in `[signifikans]` config), plan-01 niche duel re-adjudication
      (C.3's tagged verdict), and the live follow-up report. Bootstrap before
      DM for readability/testability.
- [ ] **D.4 (§6.4)** Provenance: datakontroll row / manifest marks the test
      window as `latest-months`/preliminary; optional corrected-archive
      refresh job that re-reports the locked test window when SMHI publishes
      it (report-only). Caveats added to every test-based table in
      `docs/evaluation.md`.
- [ ] **D.5** Explicit decision on `trim_prediktioner_log`: trimming unrealized
      predictions at challenger switch shrinks significance samples — decide
      trim-at-realized-age vs status quo; record in `docs/decisions.md`.
- [ ] **D.6** Tests (from the plan's list): per-line gate matrix (each
      stale/fresh combination), high-PM propagation + threshold table fixture,
      bootstrap determinism (identical series → p ≈ 1; shifted → p < 0.05),
      n-guard, live-summary threshold dedup, provenance flag.

**Exit D:** every between-model difference reports n, test used, and p (or an
explicit "not yet testable" label); all three lines retrain on their own
7-day clocks; test-window provenance explicit; tests green.

## Stage E — Series closure (docs + final run)

- [ ] Full suite green + one full pipeline run under the new gates (per the
      roadmap's working rules: fixed `random_state`, pinned sklearn, docs
      updated at every step).
- [ ] Update [00-overview.md](00-overview.md) status table, the
      [implementation-roadmap.md](implementation-roadmap.md) checklists, and
      the "Original limitation verification" table (the only rows that
      remain open after this series: train–serve skew (accepted) and
      scale-out (external)).
- [ ] Register any new config keys (`[signifikans]` α, cache path, provenance
      manifest path) in the roadmap's scale-out coexistence list so the
      plan-07 agent can carry them into `for_station_horisont`.
- [ ] Revisit the constraint from 00-overview.md: "not suitable for health
      decisions until plans 01/06 measurably improve peak performance" —
      with D's High-PM + significance numbers, the team can re-evaluate it.

## Why this order

1. **A before B:** the cache bug would otherwise delete B's evidence (the
   same trap the 20:38 first run fell into).
2. **B as one gate:** the forced duel is operationally identical for
   05/01/02 — one run, three verdicts, no duplicate operations.
3. **C before D:** plan 06's significance machinery re-adjudicates C.3's
   close verdict and plugs into the composite margins — best run once real
   numbers exist.
4. **D last:** the only remaining code phase and fully self-contained;
   plan 04 is done and cannot conflict.

## Constraints carried over (unchanged)

- dropna policy; Champion/Challenger tie rule (old champion retained);
  dynamic data periods; 2026-11-15 hygiene integrations
  (`stada_models_dir` 3 feeds, per-era `trim_prediktioner_log`, 30-day
  prognos retention); SMHI CC BY 4.0 credit; no health-decision claims.
- Plan-07 agent ownership: `main.py` loop structure, `src/config.py` loading
  machinery, station/path templating — additive changes only.
