# Open Transit Delay – ITSC Delay Pipeline

A research-oriented Python pipeline for reconstructing bus arrival times from vehicle position (VP) data, matching vehicles via anchor stops, and generating publication-ready delay and layover visualizations.

This project supports delay analysis, timetable robustness evaluation, and reproducible research workflows in public transport systems.

> **Attribution.** This reproducibility package was re-founded with the support of **Claude Opus 4.8 (Anthropic)**: the vehicle-matching and delay-reconstruction parameters were brought in line with the calibrated operational pipeline behind the paper, and every parameter is documented in [`PARAMETERS.md`](PARAMETERS.md).

> **Pseudonymized sample data.** The shipped data carries pseudonymized identifiers only — `vehicle_id` → `V0001…` and operator `unternehmen` → `Operator A…`. All coordinates, times, geometry and schedule structure are real and unchanged, so results are fully reproducible; only the labels are anonymized. See [`PARAMETERS.md`](PARAMETERS.md#pseudonymization-of-the-shipped-data) and `scripts/pseudonymize_data.py`.

---

## Overview

This software reconstructs actual arrival times for scheduled bus services by combining:

- Scheduled stop data (SOLL)
- Vehicle position data (VP)
- Route segment geometries (via OSRM or GeoJSON)

It includes automated vehicle identification, probabilistic map matching, robust interpolation strategies, and high-resolution figure generation for academic publications.

---

## Core Features

### 1. Anchor-Based Vehicle Matching

Automatic identification of the correct vehicle for a scheduled service day using spatial-temporal anchor matching at start and end stops.

Vehicle candidates are ranked using weighted distance and time deviations.

---

### 2. Arrival Time Reconstruction

Arrival times are reconstructed using a three-stage approach:

1. Strict spatial-temporal anchor detection  
2. Interpolation between anchor stops  
3. Trajectory-based interpolation or bounded extrapolation  

Optional two-stage edge gap filling improves robustness at first and last stops.

All timestamps are internally normalized to UTC.

---

### 3. Likelihood-Based Map Matching

Vehicle positions are projected onto scheduled route segments using a Gaussian likelihood model in metric space.

---

### 4. OSRM Integration (Optional)

Route segments between scheduled stops can be generated dynamically via OSRM.

Alternatively, precomputed route segments can be loaded from a GeoJSON file.

---

### 5. Publication-Ready Visualizations

The pipeline generates high-resolution TIFF figures:

1. Boxplots of absolute delay by scheduled hour  
2. Split violin plots of scheduled vs actual layover times  
3. Heatmap of delay along a vehicle schedule  
4. Deviation-from-schedule plots with buffer markers  
5. Actual travel time curves (optionally capped)

All figures are fully configurable via `config.yaml`.

---

## Project Structure
src/itsc_delay_pipeline/
cli.py # Typer CLI entry point
config.py # YAML config loader
io.py # Input readers (CSV, GeoJSON)
matching.py # Core matching and interpolation logic
osrm.py # OSRM segment builder
pipeline.py # End-to-end processing pipeline
visualize_results.py # Paper-style plots
viz.py # Interactive Folium map export


---

## Installation

Python 3.10 or newer is recommended.

Clone the repository and install in editable mode:
git clone https://github.com/
<your-user>/open-transit-delay.git
cd open-transit-delay
pip install -e .


Main dependencies:

- pandas
- geopandas
- shapely
- numpy
- matplotlib
- seaborn
- folium
- requests
- typer
- pyyaml

---

## Usage

All commands use a YAML configuration file.

### Run pipeline
Main dependencies:

- pandas
- geopandas
- shapely
- numpy
- matplotlib
- seaborn
- folium
- requests
- typer
- pyyaml

---

## Usage

All commands use a YAML configuration file.

### Run pipeline


itsc-delay run -c config.yaml


Override specific days:


itsc-delay run -c config.yaml --tag 2026-01-08
itsc-delay run -c config.yaml --tags 2026-01-08,2026-01-15


---

### Generate publication figures


itsc-delay visualize-results -c config.yaml


---

### Run pipeline and figures


itsc-delay run-all -c config.yaml


---

## Methodological Outline

### Anchor Vehicle Matching

Start and end stops of scheduled trips serve as spatial-temporal anchors.  
Candidate vehicle IDs are evaluated based on:

- Temporal deviation  
- Spatial distance  
- Weighted normalized scoring  

The vehicle with the highest anchor consistency across the service day is selected.

---

### Arrival Time Reconstruction Logic

For each stop:

- If a strict anchor exists → use anchor timestamp  
- If two anchors exist → interpolate in time–distance domain  
- Otherwise → interpolate or extrapolate along projected trajectory  

---

### Delay Computation

Delay is computed per stop as:

    delay_s = ankunft_ist − ankunft_soll   (actual − scheduled arrival, seconds; negative = early)

It is written as the `delay_s` column of the arrivals CSV.

---

### Assignment Confidence

Each vehicle assignment carries a size-aware **confidence gate**. The relative
margin `mrel = (top_hits − second_hits) / top_hits` separates correct from wrong
assignments (median 0.92 vs 0.29 against ground truth); a `confident` flag is
written to `vehicle_assignment_<umlauf>_<tag>.csv`. Downstream analyses should
keep only `confident = 1` rows. Thresholds and the calibration are in
[`PARAMETERS.md`](PARAMETERS.md#confidence-gate).

---

## Output

For each `(umlauf_id, tag)` combination:

- `stops_arrivals_anchor_interp_<umlauf>_<tag>.csv` — per-stop scheduled/actual arrival + `delay_s`
- `stops_arrivals_anchor_interp_<umlauf>_<tag>.html`
- `vehicle_assignment_<umlauf>_<tag>.csv` — matched vehicle + confidence gate (`mrel`, `confident`)
- `anchor_candidates_<umlauf>_<tag>.csv` — anchor diagnostics
- `anchor_vehicle_ranking_<umlauf>_<tag>.csv` — vehicle vote ranking
- High-resolution TIFF figures

---

## Configuration

Configured via `config.yaml`.

Important sections:

- `paths.*`
- `run.tag`
- `run.umlauf_id`
- `arrivals.*`
- `anchor_vehicle_match.*`
- `visualize_results.*`
- `osrm.*`

---

## OSRM Support

> **Reproduction prerequisite.** Segment geometries between stops come from an OSRM routing server (driving profile) for the study region. To reproduce, run your own OSRM instance and set `osrm.base_url` (the shipped config points at `http://localhost:5000`), **or** provide a precomputed `fallback_segments_geojson` with the schema below and leave `osrm.base_url` empty. Everything else (matching, delay reconstruction, figures) then runs offline on the shipped pseudonymized data.

If `osrm.base_url` is defined in the configuration, route segments are generated dynamically.

Otherwise, a fallback GeoJSON must contain:


frt_fid
edge_idx
cum_start_m
cum_end_m
geometry