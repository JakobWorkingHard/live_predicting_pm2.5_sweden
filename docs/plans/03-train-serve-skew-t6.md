# Plan 03: Train–serve skew of the T+6 line — ACCEPTED LIMITATION (no action)

Part of the limitation-fix series for Predicting Air Quality
(PM2.5, Malmö Dalaplan). Status decided with the team 2026-11-15:
**the train–serve skew of the förskjuten (T+6) line is an accepted,
intentional design trade-off. No work is planned on this limitation.**

## Background (from the report)

The förskjutna line is trained and historically evaluated with
*observed* weather at T+6 ("facit"/oracle weather), while live mode
feeds it SMHI's point forecast for the måltid. The report flagged this
as a train–serve skew: the T+6 line's edge over the vanliga line was
only 0.009 µg/m³ MAE *with perfect future weather*, and forecast error
in drift will likely eat that margin.

## Why it is fine (team decision, 2026-11-15)

- The design is deliberate and documented (config.toml `[model.forskjuten]`,
  docs/evaluation.md): the förskjuna prognosen answers the conditional
  question *"what PM2.5 is expected at T+6 **if** the forecasted
  weather materializes?"* — an analytically interesting quantity in its
  own right, rather than a degraded attempt at the unconditional one.
- The vanliga line — the recommended champion for general forecasts —
  does not depend on forecasts in any slot, so production-relevant
  forecasts are unaffected by the skew.
- The live pipeline already logs the forecast actually used
  (`prognos_referenstid` in `prognoser/prediktioner_live.csv`, plus the
  per-run forecast slot log message), and `live.py --uppföljning`
  measures the förskjuna line's real drift performance against
  realized measurements. Drift performance is therefore *observable*
  even without a hindcast study.

## What we keep doing (already implemented, no change)

- Keep archiving forecasts in `prognoser/` (raw JSON + 24 h CSV) and
  the PM2.5 archive; keep `live.py --uppföljning` closing the loop.
- Keep the caveat in docs/evaluation.md that the förskjuna line's
  historical results are oracle-based and must not be read as verified
  drift performance.

## Formerly proposed work, now dropped

- ~~Hindcast evaluation on the archived forecasts~~ — dropped.
- ~~Combined Δ-weather model (weather at T + forecast at T+6 + PM2.5
  lags T−1/T−2)~~ — dropped for now (may be revisited if the förskjuna
  line ever becomes the recommended production model).
- ~~Raising `prognos_bibehall_dagar` for hindcast purposes~~ — not
  needed; current 30-day retention stands.

## Revisit trigger (only if)

- The förskjuna line starts winning champion duels and becomes the
  recommended model for live use, or
- follow-up (`--uppföljning`) shows a persistent large gap between the
  förskjuna line's live MAE and its oracle evaluation MAE.
