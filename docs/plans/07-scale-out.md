# Plan 07: Scale out to all Swedish air-quality stations and horizons 1–24 h

Part of the limitation-fix series for Predicting Air Quality
(PM2.5, Malmö Dalaplan). Source: report §7 ("Individuell slutsats") —
the author's stated natural next step: since the architecture was built
around `config.toml` and `orchestrator.py` to grow, "låta main.py loopa
över samtliga mätstationer med tillhörande väderstationer och träna
nischade modeller för varje horisont från 1 till 24 timmar framåt".
This is a long-term extension: start only after plans 02–06.

## Limitation

The current system serves exactly one station (Malmö Dalaplan,
tidsserie 6247) at one horizon (T+6). The report frames this as the
main scope bound: street-canyon measurements at one station do not
represent all of Malmö; other pollutants, stations and horizons were
deliberately cut from the MVP.

## Current state (verified 2026-11-15)

- `config.toml` hard-codes a single `[station.pm25]` (Malmö Dalaplan,
  id 6247) and one `[station.vader]` (Malmö A, id 52350) pair; paths,
  champion logs and the live prognos URL are all built from that pair
  via `src/config.py` (`skapa_testvagar`, placeholders like
  `{pm25_tidsserie_id}`).
- `orchestrator.py` runs a single-tenant pipeline: fetch/merge →
  process → vanlig line → förskjuten line → head-to-head.
- `live.py`/`predict_pipeline.py` predicate for the single station's
  coordinates (prognos URL built with the weather station's lat/lon).
- Models/champions live in one flat `models/` with
  `stada_models_dir` keyed on two champion logs (vanlig, forskjuten_
  prefixed files).

## Goal

A multi-station, multi-horizon loop over the existing architecture:
for each Swedish PM2.5 station with acceptable data coverage, and each
horizon 1–24 h, train and serve per-(station, horizon) niche models
with their own champions, without rewriting the pipeline per station.

## Implementation steps

1. **Station registry** — replace `[station.pm25]`/`[station.vader]`
   with an array section (or a `stations.toml` listing):
   `[[station]]` with the air station (name, tidsserie_id, lat/lon),
   its paired weather station (id, distance), and an enabled flag.
   - Bootstrap script `src/stationskatalog.py`: query the 52North API
     station list, filter active stations measuring PM2.5, and pair
     each with the geographically nearest SMHI CORE weather station
     (report distance; flag pairs > ~50 km for human review).
   - Config defaults to only Malmö Dalaplan enabled; others opt-in.
2. **Config templating** — `src/config.py` gains
   `ProjectConfig.for_station_horisont(station, horisont)` producing
   fully namespaced paths:
   `models/<station_id>/h<6>/…`, `logs/<station_id>/champion_vanlig_h6.csv`,
   `data/processed/<station_id>/…`, `prognoser/<station_id>/…`.
   `skapa_testvagar` and the `[paths.filenames]` placeholders extend
   with `{station_id}` and use `[model].horisont` per loop iteration.
3. **Horizon loop** — hoist `[model].horisont` into the loop
   (1…24, or a configured subset to start, e.g. {1, 6, 12, 24}):
   `build_model_registry`, `ersatt_vader_med_maltidsvader(df, …, timmar=h)`,
   `add_time_features(df, horisont=h)` and target `shift(-h)` are
   already parameterized — pass h through `run_model_line`.
   - Champion name suffixes already carry the horizon
     (`f" vanlig (T+{cfg.horisont})"`) and freshness is suffix-locked —
     this protects per-horizon staleness for free once cfg is
     per-horizon.
4. **Orchestrator restructure** — top-level loop in
   `main.py`/`orchestrator.py`:
   ```
   for station in aktiva_stationer(cfg):
       for h in horisonter(cfg):
           orchestration(cfg.for_station_horisont(station, h))
   ```
   with per-(station, horizon) try/except so one bad station never
   kills the batch; a roll-up summary table per run (station × horizon
   × champion × composite score) appended to
   `logs/skalningslogg.csv`.
5. **Live pipeline** — `kor_live_prediktion(cfg)` per station/horizon;
   the prognos URL must use the *air station's* coordinates for Malmö's
   setup distinction (today it uses the weather station's lat/lon —
   verify per station; document the choice in docs/data.md).
   - `stada_prognos_dir`/`stada_models_dir` operate in already
     namespaced directories unchanged.
6. **Scheduling & budget** — a champion-freshness driven scheduler:
   only (station, horizon) pairs with stale champions run the full
   loop; others skip. Stagger (e.g. N stations per night) since
   24 horizons × K stations is expensive. Consider CLI `main.py
   --station <id> --horisont <h>` for targeted runs during rollout.

## Affected files

- `config.toml` / `src/config.py` — station array, per-target config
  factory, namespaced paths
- `src/stationskatalog.py` (new) — station discovery & pairing
- `src/orchestrator.py`, `main.py` — station/horizon loops, roll-up
- `src/data_pipeline.py` — namespaced fetch/process (little change;
  paths carry the namespace)
- `src/predict_pipeline.py` — per-station live, coordinates fix
- `src/io.py` — log paths, `stada_models_dir` with namespaced feeds
- `tests/test_config.py` (new), existing path tests updated

## Tests

- `for_station_horisont` produces disjoint paths for two stations and
  two horizons; placeholders all resolve; champion log names differ.
- Station catalog: fixture API response → correct filtering, nearest-
  weather pairing and distance computation.
- Loop isolation: one station raising in fetch does not abort
  subsequent stations (stub failure).
- Freshness: a stale (station=2, h=12) champion retrains while a fresh
  (station=1, h=6) is skipped — proof the suffix/freshness mechanism
  composes.

## Acceptance criteria

- One config change enables a second real station end-to-end (fetch →
  champions → live predictions → follow-up), with Malmö Dalaplan
  results byte-identical to before (namespace `dalaplan` reproduces
  current paths or a documented one-time migration).
- At least 4 horizons (1, 6, 12, 24) running for Dalaplan as the
  reference deployment before more stations are enabled.
- Roll-up log gives a single-screen status of the whole fleet.

## Risks & notes

- **Cost/time explodes linearly** in stations × horizons × search
  (plan 05): land 02–06 first; keep per-run budgets in config and
  prefer the fresh-skip scheduler.
- Many small stations have *worse* data coverage → dropna loss and
  missingness features (plan 04) matter more; add a coverage gate
  (e.g. ≥ 80 % complete hours) to the station catalog.
- Shorter horizons (h=1–3) re-rank models (persistence is very hard to
  beat) — expect different champions per horizon; that is the point of
  niche models, and the composite formula (plan 02) per horizon keeps
  choices honest.
- API politeness: rate-limit the discovery and fetch loops against
  SMHI/Naturvårdsverket (CC0/CC BY, but be a good citizen); cache raw
  fetches like today.
