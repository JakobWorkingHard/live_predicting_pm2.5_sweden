# spec.md — Live PM2.5 Prediction for Sweden (multi-station)

> Specification of the project `LIVE_predicting_PM2.5_SWEDEN` (repo name: `predicting_air_quality`), compiled from the codebase, `README.md`, `config.toml` and `docs/`.
>
> **Reflects the codebase as of 2026-11-15** (after plan 02 composite/validation-duel + locked test snapshot, plan 05 hyperparameter search, plan 07 multi-station scale-out, plan 01 nisch line, and plan 04 missing-data observability: gap-pattern logs, machine-readable live skip log, no-crash per-line degradation).
>
> **Language policy:** the long-term goal is to migrate the entire project (identifiers, docs, logs, comments) to English for token efficiency. This spec is written in English throughout. Swedish identifiers still present in the code are referenced where needed for accuracy, and Section 11 maps them to proposed English equivalents so this spec can serve as the baseline for the migration.

## 1. Overview

**What it does:** Predicts the PM2.5 concentration **6 hours ahead** for every discovered **station pair** — a PM2.5 station paired with its nearest qualifying SMHI weather station within 10 km — using the latest PM2.5 measurement, weather data, and time-of-day features. Stations are discovered automatically from SMHI's APIs and cached for 7 days (≈18 pairs as of 2026-11-15; Malmö Dalaplan remains the worked example throughout). It is a full ML pipeline with continuous training, automatic champion-model selection, live prediction, and follow-up of live predictions against actual measurements. Missing data is never imputed: its *pattern* is logged in batch mode (gap/block statistics), and in live mode every skip is machine-logged with an explicit reason and missing data can never crash a run (all since 2026-11-15, plan 04).

**Status:** University/course project and demonstration of an ML pipeline with continuous training — not a production service. Current champion models (SVR) achieve ~11–12 % lower MAE than a persistence baseline on test data (~0.2 µg/m³), but systematically underestimate high PM2.5 levels, so it must **not** be used as a warning system or for health decisions.

**Out of scope:** other pollutants (PM10, NO₂), horizons other than 6 hours, warning services, apps, or health advice.

## 2. Data sources

All APIs are open; no API keys or environment variables are required.

| Data | Provider / steward | API | Station |
|---|---|---|---|
| PM2.5 hourly (µg/m³) | Naturvårdsverket (SMHI is data host; measurements by Malmö stad among others) | `https://datavardluft.smhi.se/52North/api` — catalogue `/timeseries` (filtered on PM2.5 phenomenon **6001**), data `/timeseries/{id}/getData` | **auto-discovered**; per station the series with the latest `lastValue` (e.g. Malmö Dalaplan, street canyon, time series **6247**) |
| Weather observations (hourly) | SMHI | `https://opendata-download-metobs.smhi.se/api/version/1.0`; station catalogues via `parameter/{id}.json` — a station qualifies only if it is active for **all five** parameters | **auto-paired**: the nearest qualifying station within 10 km of the PM2.5 station (e.g. Malmö A, id **52350**, ~4 km from Dalaplan); a weather station may serve several PM2.5 stations |
| Weather forecasts (live only) | SMHI | punktprognos (`snow1g`), URL formatted per pair from its weather station's coordinates | — |

**Weather parameters** (SMHI parameter numbers): Lufttemperatur (1, °C), Vindriktning (3, degrees), Vindhastighet (4, m/s), Relativ luftfuktighet (6, %), Nederbördsmängd (7, mm).

**Data quality split:** training/validation use the quality-controlled archive (`corrected-archive`); test and live use uncontrolled real-time data (`latest-months`) that SMHI may correct retroactively.

**Licenses:** code MIT; SMHI data CC BY 4.0; air-quality data CC0 1.0.

## 3. Dynamic time periods

Periods are recomputed **per station pair** on every run (`berakna_tranings_och_valideringsperiod`, called from `bygg_station_config` in `src/config.py`), driven by the latest `corrected-archive` timestamp of the **pair's weather station** and the pair's rolling train start:

| Period | Rule | Example (Malmö pair, run 2026-11-15) |
|---|---|---|
| Training | Since 2026-11-15: `max(series first value, today − 4 years)` (`[time] train_rullande_ar = 4`) → day before validation start. Younger series start at their first value; older series are capped at a 4-year rolling window. | 2023-05-26 – 2026-04-30 (series start is younger than the window) |
| Validation | Last 6 full calendar months before the month where `corrected-archive` ends (5 months if 6 unavailable) | 2026-05-01 – 2026-10-31 |
| Test | First available hour in `latest-months` after validation end → now − 4 h (rolling, re-fetched every run — feeds datakontroll and the processed build only) | 2026-11-01 – run date |

The **evaluation** test set is not the rolling period: since plan 02 (2026-11-15) each pair has a **locked test snapshot** `data/test/last_{id}` (frozen once via `python main.py --ladda-test`, with a provenance manifest marking the `latest-months` origin as preliminary) that is used exclusively for the independent head-to-head report. A re-snapshot is at most a once-per-major-version decision, always recorded in the decision log.

## 4. Pipeline architecture

### 4.1 Entry points

| Command | What it runs |
|---|---|
| `python main.py [--station <name>] [--ladda-test] [--ersatt]` | Load base config → **discover station pairs** (fresh 7-day cache, otherwise full API discovery) → run the full pipeline for **each pair** (`src.orchestrator.orchestration`) with per-pair error isolation → print an end-of-run summary table (OK/FEL per pair). `--station` runs only pairs whose PM2.5 or weather station name matches the substring; `--ladda-test` freezes the locked test snapshot per pair; `--ersatt` replaces an existing snapshot |
| `python live.py [--uppföljning] [--station <name>]` | Same loop for live operation: one prediction per pair with both champion models (`kor_live_prediktion`), or reconcile logged predictions against actual PM2.5 measurements (`folj_uppa`) |
| `pytest` | Test suite in `tests/` |

Nothing happens at import time: base config parsing is pure TOML, and all API discovery, fetching and training run **only inside** `main()`/`live()`.

### 4.2 Configuration

**Two-layer configuration (since 2026-11-15).** `config.toml` is parsed once by `ladda_baskonfig` into a **`BasConfig`** (pure TOML, no API calls): API URLs, model settings, `[stations]` discovery thresholds, and *path templates* with `{pm25_tidsserie_id}` placeholders. For each station pair, **`bygg_station_config(bas, par)`** then builds a `ProjectConfig` — the same dataclass the whole pipeline has always consumed — with every pair-specific artifact namespaced by the PM2.5 series id: champion/nisch/datakontroll/comparison logs (`logs/champion_vanlig_{id}.csv`, …), `models/{id}/`, `prognoser/{id}/` (incl. `prediktioner_live.csv`), `data/test/last_{id}`, the drop-analysis log `logs/bortfallsanalys_{id}.csv` and the skip log `logs/live_skippad_{id}.csv` (both since plan 04), validated/processed/förskjuten dataset filenames, and the forecast URL formatted from the pair's weather-station coordinates. Downstream pipeline code is unaware of the multi-station loop. Otherwise unchanged: horizon (`[model] horisont = 6`), features, and live settings (max age, idempotency, forecast parameter mapping, archive retention windows `prognos_bibehall_dagar` / `pm25_arkiv_bibehall_dagar`). Model selection is configured through the composite weights (`[model.composite]`, plan 02), the hyperparameter search (`[model.hparam_search]`, plan 05) and the High-PM threshold rule (`[model.high_pm] threshold_rule`, plan 01).

### 4.3 Station discovery & pairing (`src/stations.py`)

Runs **only inside** `main()`/`live()` via `hamta_stationpar(bas)` — never at import time. The pair list is cached in `data/stationer/stationspar_v1.csv` (human-readable; freshness `[stations].cache_giltig_dagar` = 7 days). If the cache is missing or stale, a full discovery runs:

1. **Air-quality stations:** bulk fetch of the 52North `/timeseries` catalogue, filtered on the PM2.5 phenomenon (**6001**). A station can have several PM2.5 series (e.g. Dalaplan's retired series 126 and current **6247**); per station, the series with the latest `lastValue` is selected. Series metadata (~163 calls, slow API — fetched with an 8-thread pool plus repair passes) also yields each series' first/last value.
2. **Active-reporting gate:** the selected series must have a value within `[stations].max_serie_alder_dagar` (14 days); otherwise the station is considered dormant and skipped.
3. **Weather stations:** `parameter/{id}.json` catalogues for the five parameters are intersected; only stations active in **all five** qualify, with `data_fran` = the latest start date among the parameters.
4. **Pairing:** each PM2.5 station (the pairing "main") takes its **nearest** qualifying weather station within `[stations].max_km` (10 km, haversine). A weather station may serve several PM2.5 stations (observed 2026-11-15: Stockholm-Observatoriekullen serves ~5 pairs, Malmö A 2).
5. **Window-coverage gate:** the weather station's data must exist since the training window start (the rolling 4-year cap or the series start).
6. **Data-sufficiency gate:** at least `[stations].min_traningsmanader` (24) full months between the series start and today minus the reserved validation months (6) — stations enter/leave the loop automatically as they age in or go dormant.

Failure behaviour: if discovery fails (e.g. the air-quality API is down — observed 503s and ~12 s/call on 2026-11-15) and *any* cache exists, the stale cache is used with a warning; a corrupt cache forces re-discovery; if the first discovery finds no pair at all, the run aborts.

### 4.4 Orchestration (`src/orchestrator.py`)

1. **Freshness gate:** if the *vanlig* champion was trained within `antal_dagar_sedan` (7 days), the whole pipeline is skipped and the latest comparison is displayed instead. Fail-safe: missing log/row/timestamp ⇒ not fresh ⇒ full training.
2. Fetch + merge test data, then train and validation data (train/val from `corrected-archive`).
3. Build the six processed datasets (train/val/test × vanlig/förskjuten).
4. Run the *vanlig* model line and the *förskjuten* model line.
5. Head-to-head comparison of both champions + persistence baseline against the pair's **locked test snapshot** (report-only since plan 02; skipped fail-safe when the snapshot is missing), logged per pair to `logs/jamforelse_champion_{pm25_tidsserie_id}.csv`.
6. **Model-dir cleaning (both branches):** `stada_models_dir` keeps only the current champion + one fallback file per line, deleting every other `.joblib`/`.json` pair in the pair's `models/{pm25_tidsserie_id}/`. Fail-safe: a missing or unreadable champion log protects that line's files (a missing *vanlig* log protects everything) — see 4.12.

### 4.5 Data extraction (`src/extracting_data.py`)

- `hamta_pm25`: fetches PM2.5 month by month (large queries are slow/fragile); Swedish timezone; deduplication at month boundaries; end date inclusive.
- `hamta_vader_arkiv` / `vader_data`: fetches the full `corrected-archive` per parameter, then slices the period in code.
- `hamta_testperiod` / `hamta_test_vader`: test data from `latest-months`.
- Live fetchers: `hamta_prognos` (point forecast JSON, English parameter keys mapped to Swedish column names via `[live].prognos_parametrar`), `hamta_senaste_pm25` (last 48 h by default; archives **one file per whole hour** — `pm25_<date>T<hh>00.csv` — so repeated live runs within an hour reuse the hour's fetch), `hamta_senaste_vaderobs`.
- `stada_prognos_dir`: retention cleaning run at the end of every live run — forecast archive files older than `[live].prognos_bibehall_dagar` (30 d) and pm25 fetch files older than `[live].pm25_arkiv_bibehall_dagar` (7 d) are deleted. Ages are parsed strictly from the filenames (both current and the old second-resolution pm25 format), so no files are read and foreign files are never touched.
- `hamta_med_retry`: shared retry wrapper for HTTP calls.
- Raw existing files act as a cache: data is never re-fetched if the CSV already exists (`generate_csv` is "create only if missing"; processed files are never overwritten).

### 4.6 Merge & validation (`src/io.py`, `src/validate_process.py`)

- `merge_df_and_generate_csv`: outer join of PM2.5 and weather on timestamp → `data/merged/`.
- `validate_data`:
  1. parse timestamps to real Swedish-time datetimes (`Europe/Stockholm`);
  2. sort chronologically;
  3. abort on duplicate timestamps;
  4. insert empty rows so there is exactly one row per hour;
  5. replace invalid values with missing: negative PM2.5, wind direction outside 0–360°, wind speed < 0, humidity outside 0–100 %, negative precipitation;
  6. map wind direction 360° → 0°.
- **No imputation/interpolation ever** — interpolation was deliberately removed to avoid data leakage. Rows with missing values (target or features) are dropped after target creation.
- **Missing-data pattern logging (plan 04, since 2026-11-15):** because evidence disappears at `dropna`, the *pattern* of missing values is compiled **before** the drop, on the validated hour-complete data (`find_nan_gaps` / `sammanstall_bortfall` in `src/validate_process.py`, called from `build_processed_dataset` for all six datasets). Per model column, the datakontroll log gains two columns — `langsta_gap_<column>_h` (longest consecutive gap) and `antal_gap_over_horisont_<column>` (number of gaps ≥ horizon; 0-columns are written too so the schema is stable across runs, appended via schema migration) — plus a top-3 run-log summary. Every gap ≥ 6 h also gets its own row (column, start, end, hours, month) in the pair's `logs/bortfallsanalys_{id}.csv` (written only when such gaps exist). First pattern review (2026-11-15, numbers in `docs/data.md`): precipitation is missing in **systematic whole-day blocks** (13 exactly-24 h gaps in the training period where SMHI rejected the day), the largest shared station outage is 296–317 h (Aug 2024), and the longest PM2.5 gap is 93 h (May 2024) — far above the 17 h previously assumed.

### 4.7 Features & target (`src/features.py`)

- **Target:** PM2.5 at T + 6 h.
- **Features** (`[model.features]`): `pm25`, `Lufttemperatur`, `vind_sin`, `vind_cos` (circular encoding of wind direction; calm = 0/0), `Vindhastighet`, `Relativ luftfuktighet`, `Nederbördsmängd`, and time features for the *target* time: `timme_sin`, `timme_cos`, `helg` (weekend), `helg_timme_sin`, `helg_timme_cos`.
- `ersatt_vader_med_maltidsvader` (`src/features.py`): builds the förskjuten datasets by replacing weather columns on row T with observed values at T + horizon (before sin/cos conversion, so wind features also reflect target-time weather).

### 4.8 Two model lines

| Line | Weather input at training | Weather input in live operation |
|---|---|---|
| **vanlig** | Observed weather at T | Latest weather observation at T_obs |
| **förskjuten (T+6)** | Observed weather at T + 6 (idealized "target-time weather") | SMHI **weather forecast** for T_obs + 6 |

Slot alignment is identical across training and live (enforced in `bygg_live_rader`), but the förskjuten line is historically evaluated with *observed* future weather — a known train–serve skew; live performance is expected to be worse than the historical evaluation.

### 4.9 Models (`src/train_models.py`)

Model registry (all trained on both lines, `random_state = 42`):

1. **Persistence Baseline** — PM2.5 now = PM2.5 in 6 h (reference; must be beaten).
2. **Linear Regression**
3. **Ridge Regression** (StandardScaler pipeline, α = 1.0)
4. **Random Forest** (200 trees, n_jobs = −1)
5. **HistGradientBoosting**
6. **SVR** (StandardScaler pipeline, RBF kernel, C = 1.0, ε = 0.1)
7. **LightGBM**
8. **XGBoost**

The listed per-family constants above are the **library defaults**; when `[model.hparam_search].aktiverad = true` (plan 05, since 2026-11-15) they are replaced at every retraining by a systematic search: GridSearchCV with a chronological `TimeSeriesSplit(gap = horizon)` **on the training period only** (no leakage into validation/test), score `−(MAE + λ·High-PM MAE)` with `λ = lambda_high_pm` (0.5; High-PM = p90 of the training target). Linear Regression has no parameters to search; the SVR grid is capped at `svr_max_rader` most recent rows (O(n²) cost). Search results are shared between both model lines within one pipeline run, cached in a timestamped JSON beside the model files, and persisted in the champion's `.json` metadata (`hparam_sok`). A failed family search falls back to library defaults (fail-safe). First run (2026-11-15): SVR picked C = 2.0, ε = 0.4 — effect ≈ 0; the tuned-vs-default Δ report is outstanding.

**Metrics** (`src/metrics.py`): MAE, RMSE, R², Bias, plus MAE/Bias restricted to high PM2.5 (above the training data's 90th percentile, ≈ 14.4 µg/m³ — internal analysis threshold, not a health limit).

**Selection metric (plan 02, since 2026-11-15) — Validation Composite** = 0.5·MAE + 0.35·High-PM MAE + 0.15·|High-PM Bias| (lower is better; weights in `[model.composite]`). If high-PM support is missing (NaN, or < `min_antal_high` = 10 rows above the threshold) the remaining weights are renormalized with a warning; with no support at all the score degenerates to pure MAE. The composite's sub-scores are logged alongside (`score_mae`, `score_high_mae`, `score_high_bias`). Note the two thresholds that must never be confused: the composite's high-PM threshold is p90 of the training **target** (evaluation), while the nisch line's routing threshold (§4.10/§4.11, plan 01) is p90 of the training **`pm25` at T**.

### 4.10 Champion logic (`src/champion.py`, `src/model_pipeline.py`, `src/io.py`)

Per model line (`run_model_line`):

1. If the line's champion is fresh (< 7 days, matched by`(T+6)`name suffix), skip training and load the existing champion.
2. Otherwise train all models (with the hyperparameter search, §4.9); best candidate = lowest **Validation Composite** (`candidate_selection_metric`, plan 02).
3. If the baseline wins, no champion is saved.
4. Otherwise the candidate duels the previous champion on **validation data**, decided by the same composite score (strictly better wins; ties keep the old champion; a win saves the challenger). When a new champion is saved, the live-prediction log is trimmed to the new champion's era (`trim_prediktioner_log`, see 4.11/4.12).
5. Winner logged (`append_champion_log`) with metric, score, composite sub-scores and model path to the pair's `logs/champion_vanlig_{pm25_tidsserie_id}.csv` / `logs/champion_forskjuten_{pm25_tidsserie_id}.csv`; model saved as `.joblib` + metadata `.json` (incl. won hyperparameters, composite weights, evaluation threshold) in the pair's `models/{pm25_tidsserie_id}/` (`save_champion_model`).

**Since 2026-11-15 (plan 02) test data is never used in any automated selection** — the config keys are truthful again (`candidate_selection_metric = champion_selection_metric = "Validation Composite"`). The locked test snapshot `data/test/last_{id}` (one-time job `main.py --ladda-test`; `--ersatt` for a documented re-snapshot) feeds **only** the independent head-to-head report (orchestration step 6), which fails safe (skipped) when the snapshot is missing. Repeated duels against a rolling test period — the adaptive-overfitting risk that motivated the change — can no longer occur.

**Champion file resolution is log-driven:** `las_champion_via_sokvag` (`src/io.py`) loads the current champion through the `sokvag` column of the champion log — never by filename glob or filesystem timestamps. Both `load_latest_champion` and `hamta_champion_modell` delegate to this single implementation, so loading and pruning (4.12) always agree on which file is current. Baseline-wins log rows have an empty `sokvag` and are skipped.

**Nisch model line for high PM2.5 (plan 01, since 2026-11-15).** At retraining a third tournament is trained on rows with `pm25` at T ≥ p90 of the training `pm25` (rule `[model.high_pm] threshold_rule = "pm25_p90_training"`; persistence baseline included — a baseline win is saved as a real champion via the `PersistenceModel` wrapper so it can actually serve). The nisch winner is **adopted only if it beats the incumbent line champion strictly on High PM2.5 MAE**, compared on the same filtered validation rows (ties/rejections keep the incumbent; only adopted champions are logged in `logs/champion_nisch_{id}.csv`, so rejections never trigger the freshness clock; the duel is cheap and re-run at every pipeline run). An adopted nisch champion has its own file prefix `nisch_`, own 7-day freshness clock, own model-directory cleanup feed, and the routing threshold persisted in its `.json` metadata — the single source of truth read by the live router (§4.11).

### 4.11 Live prediction (`src/predict_pipeline.py`, run via `live.py`)

Flow in `kor_live_prediktion`:

1. Fetch forecast + latest PM2.5 + latest weather observations.
2. **Anchor point** `T_obs` = latest full hour with complete PM2.5 + all weather parameters (`bestam_ankarpunkt`).
3. **Guards, in order:** (a) no complete hour ⇒ skip; (b) `T_obs` older than `max_alder_timmar` (6 h) ⇒ skip; (c) idempotency — `T_obs` already predicted ⇒ skip (unless `[live] idempotent = false`); (d) per line: no champion ⇒ warn and skip line. **Since plan 04 (2026-11-15) every guard/skip writes exactly one machine-readable reason row** (`append_skippad_log`) to the pair's `logs/live_skippad_{pm25_tidsserie_id}.csv` — columns `timestamp, t_obs, orsak, linje, alinje, pm25_alder_h`; reason enum: `ingen_ankarpunkt`, `for_gammal_ankarpunkt`, `t_obs_redan_predikterad`, `champion_saknas:<linje>`, `prognos_saknas_maltid`, `vader_slot_ogiltig:<linje>`, `ovantat_fel:<Typ>`. A normal complete run writes nothing.
4. `bygg_live_rader` builds one model row per **available** line, with identical slot alignment as training; both slots pass `sanity_kolla_vader` (same interval rules as training validation), then wind + time features; columns exactly `cfg.features` in training order. **Missing data degrades per line instead of crashing (plan 04):** a missing forecast row for the target time no longer raises a `ValueError` that killed both lines (pre-2026-11-15) — the *vanlig* line (needs only the T_obs observation) is still built and served while the *förskjuten* line is skipped with `prognos_saknas_maltid`; a slot left incomplete by the sanity interval check (NaN) skips its line with `vader_slot_ogiltig:<linje>` — no NaN ever reaches a model. The returned dict may lack keys; step 5 predicts whatever was built.
5. **Nisch routing (plan 01), then prediction:** after loading both champions, the router (`hamta_nisch_champion` + `bestam_vanlig_champion`) hands the *vanlig* line to the nisch champion when PM2.5 at T_obs ≥ the threshold read from the champion's metadata (threshold inclusive, exactly like the nisch training row filter; no dead zone). A missing nisch champion or metadata threshold simply turns routing off with a warning (the ordinary champion serves) — never a skip. Predict with each line's (possibly routed) champion; display; append one row per predicted line to the pair's `prognoser/{pm25_tidsserie_id}/prediktioner_live.csv` (körningstid, t_obs, mål_tid, pm25_senaste, pm25_alder_h, horisont, prognos_referenstid, linje, modell, prediktion, nisch_threshold). Rows are kept **per champion era**: when a new champion replaces the old one for a line, that line's rows from earlier champions are dropped (`trim_prediktioner_log`, called from `run_model_line` on turnover; nisch and ordinary champion turnovers each keep the other era's rows).
6. Archive raw forecast JSON in full + a trimmed 24 h CSV per run in the pair's `prognoser/{pm25_tidsserie_id}/` (archive only — prediction itself reads only the forecast row at T_obs + 6).
7. `stada_prognos_dir` runs in a `finally` around the whole flow, so the forecast archive is pruned on **every** live run — also when a guard skipped the run via an early return (see 4.12).
8. In the multi-pair loop, `live.py` isolates pairs with a try/except; **since plan 04 every unexpected per-pair failure also appends one `ovantat_fel:<Type>` reason row** to that pair's skip log before the loop continues — silent generic failures are no longer possible.

**Follow-up** (`folj_uppa`, `live.py --uppföljning`): a **skip summary** (7-day counts per reason, `visa_skip_sammanstallning`) is printed first — even when nothing has realized yet, precisely when "why are there no predictions?" matters — then logged `mål_tid` is matched against actual PM2.5 measurements (last 7 days), the 10 latest realized predictions are shown row by row, and MAE/RMSE/R²/Bias aggregated per line plus persistence baseline (`sammanstall_live_uppfoljning`; nisch-era predictions appear as their own (line, model) groups).

### 4.12 Storage & retention (since 2026-11-15)

All runtime artifacts are bounded. Until this date, every one of them accumulated indefinitely (files were never deleted and logs were never rotated). Implemented in `src/io.py` (`stada_models_dir`, `trim_prediktioner_log`), `src/extracting_data.py` (`stada_prognos_dir`) and `src/logger.py`. Since the multi-station loop (2026-11-15) every bounded artifact exists once per **station pair** (~18 pairs expected), so each bound is enforced per pair; the slow logs remain append-only audit trails, now one file per pair. All append-style CSV logs (champion, comparison, datakontroll, predictions, and since plan 04 the skip and drop-analysis logs) are written through `_append_med_schema_migration` (`src/io.py`): new columns trigger a single header-rewrite so old rows stay readable — new columns never desync the file.

| Artifact | Growth (before) | Bound mechanism (now) |
|---|---|---|
| `models/{pm25_tidsserie_id}/*.joblib` + `.json` (one directory per pair) | new timestamped pair per champion crowning, never deleted | `stada_models_dir` after every orchestration run keeps the last `keep = 2` logged model files per line (current champion + 1 fallback, resolved via the `sokvag` column) and deletes every other pair; log-driven loading (`las_champion_via_sokvag`) makes deletion safe |
| `prognoser/{pm25_tidsserie_id}/prognos_<referenceTime>_utc.{json,csv}` | ≤ 2 files per forecast fetch (new reference time up to every 15 min) | deleted when older than `[live].prognos_bibehall_dagar` (30 d) |
| `prognoser/{pm25_tidsserie_id}/pm25/*.csv` | 1 file per live run (second-resolution name) | 1 file per whole hour (hour-stamped name) + deleted when older than `[live].pm25_arkiv_bibehall_dagar` (7 d); old second-format filenames are also matched |
| `prognoser/{pm25_tidsserie_id}/prediktioner_live.csv` | 2 rows per live run, unbounded | champion-era trimming (`trim_prediktioner_log`): on champion turnover for a line, that line's rows from earlier champions are dropped (other line untouched; header preserved; atomic rewrite) — accepted trade-off: unrealized rows of the old era can no longer be followed up after a turnover |
| `logs/champion_*_{pm25_tidsserie_id}.csv`, `logs/jamforelse_champion_{pm25_tidsserie_id}.csv`, `logs/datakontroll_{pm25_tidsserie_id}.csv`, `logs/bortfallsanalys_{pm25_tidsserie_id}.csv` (plan 04), `logs/live_skippad_{pm25_tidsserie_id}.csv` (plan 04) | a few rows per run per pair (skip log: ~1 row per *skipped* live attempt; drop analysis: 1 row per gap ≥ 6 h, nothing on clean data) | **deliberately unbounded** (audit trail; volume negligible — ~1 champion row per retrain and pair, duplicates when the old champion defends its title; skip-log growth is additionally bounded by the throttled live frequency, no retention needed initially) |
| `data/stationer/stationspar_v1.csv` | — (single discovery cache) | rewritten at most every `[stations].cache_giltig_dagar` (7 d); fails over to the stale copy when the APIs are down |
| `logs/project.log` | one growing file, written relative to the working directory | nightly rotation (`TimedRotatingFileHandler`, 14 daily backups), single shared file handler, absolute path anchored to the project root (no more stray `notebooks/logs/`) |

Cleanups are fail-safe: ages are read only from strict filename patterns (foreign files untouched), and missing/unreadable/corrupt logs never cause deletions.

## 5. Results (pipeline run 2026-10-23, test data)

| Model | MAE | RMSE | R² | Bias |
|---|---:|---:|---:|---:|
| Persistence baseline | 1.741 | 2.628 | 0.414 | −0.004 |
| SVR (vanlig) | 1.545 | 2.265 | 0.565 | 0.056 |
| **SVR (förskjuten, T+6)** | **1.536** | **2.231** | **0.578** | 0.103 |

At high PM2.5 levels (> 90th percentile of training data), both SVR models have large errors (MAE ≈ 6.8) and strongly negative bias (−5.8 to −6.1), and the persistence baseline actually beats all ML models at high concentrations on validation data.

**Selection update (2026-11-15).** The results above predate the composite-based selection (plan 02) and the hyperparameter search (plan 05). First verdict under the composite (0.5·MAE + 0.35·High-PM MAE + 0.15·|High-PM Bias|) on the Malmö pair: **the persistence baseline wins the candidate tournament at all tested weightings** — models have lower total MAE but underestimate high levels by ≈ 4.5 µg/m³ where the baseline is nearly unbiased, which is exactly what the formula refuses to crown. Weight sensitivity (w_high ∈ {0.2, 0.35, 0.5}) changes the best *model* but not the baseline's win. The first hyperparameter search picked SVR C = 2.0, ε = 0.4 with effect ≈ 0. Whether to accept the composite verdict, adjust weights, or await tuned models was a pending team decision as of 2026-11-15.

## 6. Project layout

```
predicting_air_quality/
├── data/            # raw/ merged/ validated/ processed/ + stationer/ (stationpair cache)
│                    # + test/last_<tidsserie_id>/ (locked snapshot, one per pair)
│                    # (all git-ignored; local cache)
├── docs/            # data.md, decisions.md, evaluation.md
├── logs/            # champion_vanlig_<id>.csv, champion_forskjuten_<id>.csv,
│                    # champion_nisch_<id>.csv, datakontroll_<id>.csv,
│                    # jamforelse_champion_<id>.csv, bortfallsanalys_<id>.csv,
│                    # live_skippad_<id>.csv — one set per station pair —
│                    # project.log (nightly rotation, 14 backups)
├── models/          # <tidsserie_id>/ per pair: champion .joblib + metadata .json
│                    # (current + 1 fallback per line)
├── notebooks/       # 1_data_quality … 5_live_prediktioner
├── prognoser/       # <tidsserie_id>/ per pair: forecast JSON + 24h CSV (30 d retention),
│                    # pm25/ (1 file per whole hour, 7 d retention),
│                    # prediktioner_live.csv (per champion era)
├── src/             # config (BasConfig + per-pair ProjectConfig factory),
│                    # stations (discovery + pairing + 7-day cache),
│                    # orchestrator, data_pipeline, extracting_data, features,
│                    # validate_process, train_models, model_pipeline, champion,
│                    # predict_pipeline, metrics, io, logger
├── tests/           # pytest: stations, champion, data_pipeline, extracting_data, io,
│                    # metrics, model_pipeline, predict_pipeline, train_models,
│                    # validate_process (gap/skip-log tests, plan 04)
├── config.toml      # single source of all parameters (incl. [stations] discovery rules)
├── main.py          # multi-station pipeline entry point (pair loop + summary table)
├── live.py          # multi-station live prediction / follow-up entry point
├── spec.md          # this document
└── requirements.txt
```

## 7. Runtime & dependencies

- **Python 3.13.7** (`.python-version`); on macOS LightGBM/XGBoost need `libomp`.
- Dependencies (`requirements.txt`): requests, pandas, matplotlib, scikit-learn, lightgbm, xgboost, joblib, ipykernel, pytest, tabulate.
- Tests: pytest with `pythonpath = .`, tests in `tests/`.

## 8. Key design decisions (from docs/decisions.md)

1. All configuration in `config.toml` (reproducible runs, no CLI flags).
2. Merged files created automatically during fetch; merge = SQL-style join on timestamp.
3. Rows with missing values are dropped entirely (never imputed) — models need complete rows and interpolation risks leakage.
4. Wind direction split into sin/cos (circular variable).
5. Week number deliberately *not* used as a feature (winter peaks are episodic, not week-stable); hour-of-day sin/cos + weekend interactions are used instead (EDA showed weekday peak ~08:00, weekend peak ~10:00).
6. Test set excludes the last 4 hours to avoid lag-related row drops.
7. Since 2026-10-20 all periods are dynamic (fixed dates removed from config); validation extended from 3 to 6 whole months on 2026-10-21.
8. Project focus is a working automated live-forecast flow, not further model tuning of high-level underestimation (recorded as a known limitation on 2026-10-19).
9. Since 2026-11-15 all runtime storage is bounded (see 4.12): champion logs are the single source of truth for which model file is current; `models/` keeps current + 1 fallback per line; live predictions are kept per champion era; forecast/pm25 archives are pruned by configurable retention windows; `project.log` rotates nightly. The slow champion/comparison logs remain append-only as an audit trail.
10. Since 2026-11-15 the project scales to all qualifying Swedish station pairs: stations are **auto-discovered** from SMHI's APIs (no station is hardcoded in `config.toml`), the PM2.5 station is the pairing main, a weather station may serve several pairs, training uses a **rolling 4-year window** (or the series' earliest data if younger — `train_start = max(series start, today − 4 years)`), and every log/model/archive artifact is namespaced per pair by the PM2.5 series id. Run control: `--station` substring filter on both entry points; per-pair error isolation so one failed pair never stops the batch.
11. Since 2026-11-15 (plan 02) both the candidate choice and the champion duel are decided by the **composite** (0.5·MAE + 0.35·High-PM MAE + 0.15·|High-PM Bias|, weights configurable) on **validation data**; test data is never used in any automated selection, the locked snapshot is report-only, and re-snapshotting is a documented version decision. The old single-metric test-based selection was removed because MAE/R² are blind to the high-level underestimation the project is trying to fix.
12. Since 2026-11-15 (plan 05) systematic hyperparameter search replaces library defaults at every retraining: GridSearchCV + chronological `TimeSeriesSplit(gap = horizon)` on the training period only, score `−(MAE + λ·High-PM MAE)`, shared between the lines, cached, persisted in champion metadata; failed family searches fall back to defaults.
13. Since 2026-11-15 (plan 01) a **nisch champion** for high PM2.5 may be adopted (trained on rows with `pm25` ≥ p90 of the training data, adopted only on strict High-PM MAE improvement on the same filtered validation rows, threshold persisted in metadata as the single source of truth) and routed in live when PM2.5 at T_obs ≥ the threshold; a baseline win in the nisch tournament is saved as a real (`PersistenceModel`) champion so it can actually serve.
14. Since 2026-11-15 (plan 04) missing data is observable end-to-end while the `dropna` policy stays unchanged (reconfirmed team decision; still no imputation): batch pattern statistics per column plus per-gap rows; in live mode every skip is exactly one machine-readable reason row, missing data degrades per line instead of crashing the pair, and the follow-up shows a per-reason skip summary — so the effective sample size can no longer silently shrink.

## 9. Limitations

- **Fixed horizon** (6 h only) across all pairs.
- **Discovery depends on SMHI API availability:** the air-quality API is slow/unreliable (503s, ~12 s/call observed); the 7-day cache softens this, and a failed discovery fails over to the stale cache. Station set changes only when the cache refreshes (≤ 7 days of lag; the discovery thresholds are fixed in `[stations]`: 10 km radius, 14-day active-reporting window, ≥ 24 months training after the validation reserve).
- **First full run is expensive:** ~18 pairs × full history fetch ≈ 1–2+ hours (data fetching dominates); subsequent runs short-circuit on the champion freshness gates and the raw-data cache.
- **High concentrations underestimated** — unsuitable as a warning system. Mitigation in place since plan 01 (nisch line adopted on strict High-PM MAE improvement and routed live above the p90 threshold) and plan 05 (high-PM-weighted search score); the actual live effect is still unverified.
- **Förskjuten line trained on "perfect" weather** (observed T+6), but serves forecasts in live mode (train–serve skew; live performance unverified without hindcasts).
- **Model differences lack significance testing** (backlog as plan 06): champion-era sample sizes are small, era trimming discards unrealized rows at turnovers, and the composite's persistence-verdict needs an explicit team decision. The per-pair skip log (plan 04) now makes the effective n measurable; the tests themselves are outstanding.
- **Test data is a report, not a selection input** (since plan 02), but remains one frozen `latest-months` snapshot — it ages until a re-snapshot is decided, and it is uncontrolled real-time data that SMHI may correct retroactively.
- **Missing data is non-random but logged** (since plan 04): whole-day precipitation blocks, shared station outages (up to 296–317 h) and long PM2.5 gaps (93 h worst) drop rows under the unchanged `dropna` policy — some weather regimes (rain days, summer weeks) are underrepresented in training and evaluation; documented with numbers in `docs/data.md`.
- **Seasonal coverage:** only ~3.5 years of history with uneven season coverage.
- **Local artifacts:** models and logs are per machine (data/ and models/ not shared), so different group members may have different champions.
- Prediction is anchored to the latest complete observation, so the actual forecast lead time can be less than 6 h if data lags.
- **Prediction log covers only the current champion era per line** (`trim_prediktioner_log`): at a champion turnover, unrealized rows from the old era are dropped, so a follow-up immediately after a turnover only covers the new champion.

## 10. Suggested future work (from docs/evaluation.md)

1. Improve predictions at high PM2.5 — **in progress since 2026-11-15**: nisch line with live routing (plan 01), high-PM-weighted hyperparameter search (plan 05) and the high-PM-aware composite selection (plan 02); remaining work is verifying the live effect and, if needed, alternative models/losses.
2. Evaluate the förskjuten line with real historical forecasts (hindcast) — **dropped 2026-11-15**: the train–serve skew is an accepted limitation (decision record in `docs/decisions.md`); revisit only if the förskjuten line becomes the recommended production model. Live drift remains observable via `live.py --uppföljning`.
3. Statistical testing / confidence intervals for model differences — **pending (plan 06)**: paired moving-block bootstrap (primary) with n-counts; the skip log (plan 04) now supplies the honest n.
4. Continuous monitoring of live performance over time — **partially done**: the follow-up aggregates per line/era and (since plan 04) prints the per-reason skip summary; trending/alerting remains future work.
5. Variable/large forecast horizons anchored on latest complete observation.

> *Former item 6 — "Extend to more measurement stations" — was completed on 2026-11-15; see §4.3.*

## 11. Terminology glossary — Swedish → English migration map

The codebase currently uses Swedish identifiers. Proposed English equivalents, to be adopted during the language migration. Code references in this spec still use the current names.

### Domain / model concepts

| Swedish (current) | Proposed English | Note |
|---|---|---|
| vanlig (modellinje) | standard line | weather observed at prediction time T |
| förskjuten (modellinje) | shifted line | weather at T + horizon; consider "target-time-weather" if explicitness is preferred |
| horisont | horizon | forecast horizon (hours) |
| måltid / mål_tid | target time | the hour the prediction refers to |
| prognos | forecast | SMHI weather forecast (distinct from the project's own *prediction*) |
| väderobservation | weather observation | |
| prediktion | prediction | the project's PM2.5 forecast output |
| champion | champion | already English |
| ankarpunkt | anchor point | latest complete hourly observation (T_obs) |

### Source files & functions

| Swedish (current) | Proposed English |
|---|---|
| `src/extracting_data.py` → `src/fetching.py` (suggested) | module rename candidate |
| `berakna_tranings_och_valideringsperiod` | `compute_train_val_period` |
| `skapa_testvagar` | `create_test_paths` |
| `hamta_med_retry` | `fetch_with_retry` |
| `hamta_pm25` | `fetch_pm25` |
| `hamta_vader_arkiv` | `fetch_weather_archive` |
| `hamta_senaste_arkivtid` | `fetch_latest_archive_time` |
| `vader_data` | `fetch_weather` |
| `hamta_testperiod` | `fetch_test_period` |
| `hamta_test_vader` | `fetch_test_weather` |
| `hamta_prognos` | `fetch_forecast` |
| `hamta_senaste_pm25` | `fetch_latest_pm25` |
| `hamta_senaste_vaderobs` | `fetch_latest_weather_obs` |
| `stada_prognos_dir` | `clean_forecast_dir` |
| `las_champion_via_sokvag` | `read_champion_via_path` |
| `stada_models_dir` | `clean_models_dir` |
| `trim_prediktioner_log` | `trim_predictions_log` |
| `_kor_live_prediktion` | `_run_live_prediction` (internal flow of `kor_live_prediktion`) |
| `ersatt_vader_med_maltidsvader` | `replace_weather_with_target_time_weather` |
| `sanity_kolla_vader` | `sanity_check_weather` |
| `kor_live_prediktion` | `run_live_prediction` |
| `folj_uppa` | `follow_up` |
| `bestam_ankarpunkt` | `determine_anchor_point` |
| `bygg_live_rader` | `build_live_rows` |
| `visa_prognosrad` | `show_forecast_row` |
| `visa_senaste_prediktioner` | `show_latest_predictions` |

### New identifiers (champion selection, nisch, observability — 2026-11-15 plans 01/02/04/05)

| Swedish (current) | Proposed English |
|---|---|
| `sammanstall_bortfall` | `compile_missing_pattern` |
| `find_nan_gaps` | `find_nan_gaps` (unchanged) |
| `append_datakontroll_log` | `append_data_check_log` |
| `append_bortfallsanalys_log` | `append_missing_pattern_log` |
| `append_skippad_log` | `append_skip_log` |
| `visa_skip_sammanstallning` | `show_skip_summary` |
| `_append_med_schema_migration` | `_append_with_schema_migration` |
| `hamta_nisch_champion` | `get_niche_champion` |
| `bestam_vanlig_champion` | `determine_standard_champion` |
| `las_champion_metadata` | `read_champion_metadata` |
| `PersistenceModel` | `PersistenceModel` (unchanged) |

### New identifiers (multi-station, since 2026-11-15)

| Swedish (current) | Proposed English |
|---|---|
| `src/stations.py` → `src/stations.py` (unchanged) | module: discovery + pairing + cache |
| `ladda_baskonfig` | `load_base_config` |
| `bygg_station_config` | `build_station_config` |
| `StationPar` / `BasConfig` / `Pm25Station` / `Vaderstation` | `StationPair` / `BaseConfig` / `Pm25Station` / `WeatherStation` |
| `hamta_stationpar` | `get_station_pairs` |
| `hamta_pm25_stationer` | `get_pm25_stations` |
| `hamta_vaderstationer` | `get_weather_stations` |
| `para_stationer` | `pair_stations` |
| `haversine_km` | `haversine_km` (unchanged) |
| `skriv_stationpar_cache` / `las_stationpar_cache` | `write_stationpair_cache` / `read_stationpair_cache` |
| `valj_stationpar` (main.py) | `select_station_pairs` |
| `_visa_sammanfattning` (main.py) | `_show_summary` |
| `_api_get` / `_datum_lokal` (stations.py) | `_api_get` / `_local_date` |

### Config keys, paths & log files

| Swedish (current) | Proposed English |
|---|---|
| `[paths] forskjuten_log` | `shifted_log` |
| `train_slut`, `validering_start`, `validering_slut` | `train_end`, `val_start`, `val_end` |
| `[model.forskjuten]` → `[model.shifted]` | section rename |
| `[model.forskjuten] vaderkolumner` | `weather_columns` |
| `antal_dagar_sedan` | `max_champion_age_days` |
| `[live] prognos_dir`, `prognos_url`, `prognos_arkiv_timmar` | `forecast_dir`, `forecast_url`, `forecast_archive_hours` |
| `[live] prognos_parametrar` | `forecast_parameters` |
| `[live] prognos_bibehall_dagar`, `pm25_arkiv_bibehall_dagar` | `forecast_retention_days`, `pm25_archive_retention_days` |
| `[live] skippad_log` (plan 04) | `skip_log` |
| `[paths] bortfallsanalys_log` (plan 04) | `missing_pattern_log` |
| `logs/champion_forskjuten.csv` | `logs/champion_shifted.csv` |
| `logs/datakontroll.csv` | `logs/data_check.csv` |
| `logs/jamforelse_champion.csv` | `logs/champion_comparison.csv` |
| `logs/live_skippad_{id}.csv` (plan 04) | `logs/live_skipped_{id}.csv` |
| `logs/bortfallsanalys_{id}.csv` (plan 04) | `logs/missing_pattern_{id}.csv` |
| skip-log CSV columns `orsak`, `alinje` | `reason`, `scope` |
| `nisch_log`, `nisch_threshold`, `champion_nisch_<id>.csv` (plan 01) | `niche_log`, `niche_threshold`, `champion_niche_<id>.csv` |
| datakontroll gap columns `langsta_gap_<kolumn>_h`, `antal_gap_over_horisont_<kolumn>` (plan 04) | `longest_gap_<column>_h`, `n_gaps_over_horizon_<column>` |
| composite keys `w_mae`, `w_high_mae`, `w_high_bias`, `min_antal_high`, `score_mae` etc. (plan 02) | already English except the CSV column prefix — translate at migration |
| `[model.hparam_search]` keys (`aktiverad`, `n_splits`, `lambda_high_pm`, `svr_max_rader`) (plan 05) | `enabled`, `n_splits`, `lambda_high_pm`, `svr_max_rows` |
| `prognoser/` → `forecasts/`, `prediktioner_live.csv` → `predictions_live.csv` | artifact names |
| CSV columns `sokvag`, `t_obs`, `mål_tid`, `pm25_senaste`, `pm25_alder_h`, `körningstid`, `linje` | `model_path`, `t_obs`, `target_time`, `pm25_latest`, `pm25_age_h`, `run_time`, `line` |
| weather column names (`Lufttemperatur`, `Vindhastighet`, …) | English meteorological names (`air_temperature`, `wind_speed`, …) — must be renamed together with the forecast-parameter mapping in `config.toml` |
| CLI flag `--uppföljning` | `--follow-up` |
| CLI flag `--station` (substring station filter, main.py & live.py) | `--station` (unchanged) |
| `[stations]` keys (`max_km`, `max_serie_alder_dagar`, `min_traningsmanader`, `stationpar_cache`, `cache_giltig_dagar`) | English discovery keys (`max_pairing_km`, `max_series_age_days`, `min_training_months`, `stationpairs_cache`, `cache_valid_days`) |
| `[time] train_rullande_ar` | `train_window_years` |
| `[station] parametrar` | `[station] parameters` (shared by all pairs) |
| `logs/champion_*_{tidsserie_id}.csv`, `models/{tidsserie_id}/`, `prognoser/{tidsserie_id}/`, `data/test/last_{tidsserie_id}` | per-pair namespaced artifact names (translate the Swedish word parts, keep the per-pair `<series_id>` suffix convention) |

### Migration notes

- **Renaming is not purely cosmetic:** several Swedish names are embedded in file paths, CSV headers, and the champion name suffixes (`" vanlig (T+6)"` / `" forskjuten (T+6)"`) used by the freshness check, so renames must be coordinated with log/artifact migrations or a compatibility read step.
- **Renaming weather columns** requires updating the training-data column names and the `[live]` forecast-parameter mapping keys simultaneously — otherwise the live slot alignment breaks.
- **Retention logic is tied to filename patterns** (since 2026-11-15): `stada_prognos_dir` parses ages from the strict patterns `prognos_<ref>_utc.*` and `pm25_<date>T<hh><mm>.csv`, and `hamta_senaste_pm25` stamps one archive file per whole hour. Renaming archive file formats or the `prognoser/` directory requires updating the regexes in `src/extracting_data.py` in the same change, or old files stop being cleaned.
- **Dataset filenames** are built from Swedish templates in `config.toml`; since `data/` is a local git-ignored cache, renaming templates only affects fresh runs.
- **Per-pair namespacing (2026-11-15):** all artifact templates now embed `{pm25_tidsserie_id}`. Renaming the Swedish identifier must update the `[paths]`/`[live]` templates *and* the filename construction in `skapa_testvagar` in the same change — otherwise station pairs overwrite each other's artifacts again. The stationpair cache (`data/stationer/stationspar_v1.csv`) is a local, git-ignored artifact like the rest of `data/`: a fresh clone simply triggers re-discovery.
- **Skip/drop-analysis logs (plan 04):** the `orsak` enum values (`prognos_saknas_maltid`, `champion_saknas:<linje>`, …) and the datakontroll gap-column names are parsed by `visa_skip_sammanstallning` and read as plain strings — translation must update the enum vocabulary in `src/predict_pipeline.py`, the glossary above and any dashboards consistently. Old log rows keep their Swedish values; the readers must tolerate both during a transition.
- **Token-efficiency payoff:** English identifiers shorten prompts and completions in AI-assisted development and make logs/docs directly reusable without translation context.
