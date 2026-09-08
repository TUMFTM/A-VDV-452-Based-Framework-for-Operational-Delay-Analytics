# Parameters

Every tunable in `config.yaml`, with its value, what it does, and why it is set
that way. The values were calibrated in the operational pipeline behind the
paper (grid searches and holdouts against operator ground truth and ~1 Hz
on-board GPS logs); the classification tags below mirror that record:

- **CALIBRATED** — grid-searched against a held-out objective.
- **BENCHMARKED** — chosen against operator ground truth.
- **MEASURED** — derived from a direct measurement.
- **STRUCTURAL / INHERITED** — fixed by the method or carried over; not swept.

> This reproducibility package was re-founded with the support of
> **Claude Opus 4.8 (Anthropic)**.

---

## Vehicle matching (`anchor_vehicle_match`)

The correct physical vehicle for a scheduled Umlauf is found by voting: each
scheduled stop is a spatio-temporal *anchor*; the nearest-in-space-and-time
vehicle-position (VP) ping at each anchor casts a vote, and the vehicle with the
most anchor hits across the day wins.

| Parameter | Value | Class | What it does / why |
|---|---|---|---|
| `all_stops` | `true` | **CALIBRATED (fix)** | Vote with **every** scheduled stop, not just first/last. First/last-only gave a 30-stop Fahrt the same weight as a 3-stop one, and a single coincidentally nearby vehicle at the two termini could win outright. Voting over all stops forces a wrong vehicle to coincide at many intermediate anchors too. |
| `anchor_dist_m` | `200.0` m | TESTED — kept | VP-anchor search radius. Grid over {100,150,200} m × {240,360,480,600} s vs ground truth; 100 m/360 s led the tuning set (95.19% vs 94.71%) but tied on holdout (98.75%, same 2/160 errors) — ranking was noise, default kept. |
| `max_dist_m` | `200.0` m | INHERITED | Hard distance cutoff; caps `anchor_dist_m`. |
| `max_dt_s` | `600` s | INHERITED | Hard time cutoff for a candidate ping. |
| `anchor_time_buf_min` | `8` min | INHERITED | Time window each side of the scheduled time when collecting candidates. |
| `k_nearest` | `10` | INHERITED | Keep the k closest candidate pings per anchor before scoring. |
| `w_dt_s`, `w_dist_m` | `1.0`, `1.0` | INHERITED | Relative weights of the time vs distance terms in the per-candidate score. |
| `dt_scale_s`, `dist_scale_m` | `60.0` s, `50.0` m | INHERITED | Normalizers so seconds and metres are comparable in the score. |

### Confidence gate

Not every assignment is trustworthy, and raw hit counts are **not** usable on
their own (they scale with Umlauf size). The discriminator is the relative
margin `mrel = (top_hits − second_hits) / top_hits`.

| Parameter | Value | Class | What it does / why |
|---|---|---|---|
| `small_umlauf_max_fahrten` | `5` | **BENCHMARKED** | Umläufe with ≤ 5 Fahrten are "small": fewer anchors to vote with, so the same relative margin means less and needs a stricter threshold. |
| `confident_mrel_small` | `0.7` | **BENCHMARKED** | Minimum `mrel` to mark a small Umlauf `confident`. |
| `confident_mrel_large` | `0.4` | **BENCHMARKED** | Minimum `mrel` for a large Umlauf. |

Against ground truth (991 Umläufe): correct assignments had a median `mrel` of
0.92, wrong ones 0.29. With these thresholds the gate keeps 86% of Umläufe /
96.4% of stops at 99.8% accuracy, excluding 96% of the errors — a single flat
threshold does measurably worse. The `confident` flag is written to
`vehicle_assignment_<umlauf>_<tag>.csv`; downstream analyses should filter on it.

---

## Likelihood projection (`arrivals`)

VP pings are projected onto the scheduled route with a Gaussian likelihood over
the perpendicular distance, collapsing 2-D positions to 1-D distance-along-route.

| Parameter | Value | Class | What it does / why |
|---|---|---|---|
| `likelihood_max_dist_m` | `50.0` m | **CALIBRATED** (was 100) | Max ping→route distance admitted to the projection. Tuning: 50 m → 23.53 s MAE, 100 m → 23.74, 150 m → 24.14; holdout 50 vs 100: paired Δ −0.28 s, better on 48/60, p = 2.9e−07. Not a coverage trade (clean stops −0.2%). |
| `sigma_dist_m` | `25.0` m | STRUCTURAL | Gaussian width of the projection weight. Sets the floor for the cutoff above: 50 m = 2σ; below ~2σ the Gaussian truncates hard enough to drop legitimate pings. Do not lower the cutoff without re-checking coverage, not just MAE. |

---

## Anchor detection (`arrivals`)

A stop gets a high-confidence "strict anchor" timestamp when a ping falls close
to it in both space and time; otherwise the arrival is interpolated.

| Parameter | Value | Class | What it does / why |
|---|---|---|---|
| `stop_anchor_dist_m` | `25.0` m | **CALIBRATED** (was 75) | Stop radius for a strict anchor. Grid {25,40,55,75,100} m × {180,360,540} s: 25 m beat 75 m at **every** dt. Holdout MAE 22.11 vs 23.37 s, paired Δ −1.52 s, better on 46/60, p = 4.3e−05; median abs 13.50 vs 15.50 s, 59/60, p = 2.4e−11. |
| `stop_anchor_max_dt_min` | `6` min | TESTED — not identified | Time tolerance for a strict anchor. Flat/non-monotone across 180–540 s, so the middle value was kept. |
| `time_gate_min` | `10` min | INHERITED | VP window per Fahrt for arrival reconstruction. |

---

## Edge-gap fill (`edge_gap_fill`)

First and last stops sit outside any interpolation interval, so they get a
dedicated widened search in two stages.

| Parameter | Value | Class | What it does / why |
|---|---|---|---|
| `only_first_last` | `true` | INHERITED | Restrict edge-gap filling to the terminal stops. |
| stage 1 (`anchor_edge_narrow`) | 3 min / 100 m | INHERITED | Narrow first pass. |
| stage 2 (`anchor_edge_wide`) | 10 min / 300 m | INHERITED | Looser second pass. This is the single largest remaining error source (extrapolation at Fahrt edges runs far higher MAE than mid-route) and is the first thing to revisit for further accuracy. |

---

## Feed-lag correction (`feed_lag`)

The GTFS-RT feed's `time` is `COALESCE(feed vehicle timestamp, scrape time)`.
Rows where `time == time_scrape` carry the feed's **position age** rather than a
real fix time and lag the true time-at-position by ~23 s; rows with a genuine
feed timestamp lag by ~1 s. Only the fallback rows are shifted back.

| Parameter | Value | Class | What it does / why |
|---|---|---|---|
| `fallback_lag_s` | `14.3` s | **MEASURED / CALIBRATED** | Seconds subtracted from fallback rows. = median ping lag 22.0 s × absorption 0.65. The absorption factor (0.65) is the share of the lag that survives anchor selection — `detect_anchors` already ranks pings by `|ts − ts_soll|` and absorbs part of it. Chosen bias-neutral: a systematic offset would propagate into every punctuality and padding statistic. |

The raw physical ping lag (~23 s) is not applied directly precisely because
anchor selection absorbs roughly half of it; applying the full lag would
over-correct and push arrivals early.

---

## Delay

`delay_s = ankunft_ist − ankunft_soll` (actual minus scheduled arrival, in
seconds), written per stop to the arrivals CSV. Negative = early.

---

## Pseudonymization of the shipped data

The sample data ships with **pseudonymized identifiers** (see
`scripts/pseudonymize_data.py`). Two direct identifiers are replaced by
deterministic, stable pseudonyms; everything else (coordinates, times,
geometry, `trip_id`/`route_id`/`entity_id`, and the `fremdunternehmer`
true/false flag) is untouched:

| Field | File | Pseudonym |
|---|---|---|
| `vehicle_id` | `data/vp.csv` | `V0001`, `V0002`, … (sorted unique) |
| `unternehmen` | `data/soll_stops.csv` | `Operator A`, `Operator B`, … |

The mapping is a bijection on the identifier space, so vehicle matching and
every downstream statistic are unchanged — only the labels differ. The
real→pseudonym mapping is written to a gitignored file and is **not** part of
the published package.
