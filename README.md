# DGO-DSS Sandwich Detector (Double Slit Simulation)

A minimal discrete-event toy model testing whether interference-like suppression
can emerge from **local timing and capacity constraints alone** — without complex
amplitudes, wave equations, or detector phase memory.

This repository accompanies the sandwich-detector experiment described in the
associated DGO paper.

---

## Conceptual motivation

In standard wave physics, a two-slit experiment produces an interference pattern
because wave amplitudes combine constructively or destructively before detection.

This model asks a different question:

> Can a structured suppression pattern appear across a detector strip using only
> discrete signals, local cyclic phase gating, and a capacity limit of one signal
> per detector per tick?

Within this toy model, the answer is yes. The strength and spatial structure of
the suppression depend on how the lower transport row is routed to the detector strip.

---

## Physical ontology of the toy model

All behaviour in this simulation follows from four rules. There are no wave
amplitudes, no superposition, no complex numbers, and no detector phase memory.

| Concept | Rule |
|---|---|
| **signal** | A unit event carrying only a row-provenance tag (`upper` or `lower`) |
| **recv\_open** | A transport node accepts a signal when `theta % m == 0` |
| **send\_open** | A transport node forwards a buffered signal when `theta % m == tau0` |
| **buffer** | A signal waits exactly `tau0` ticks between reception and forwarding |
| **capacity** | A detector accepts at most `capacity_detector` signals per tick; excess is blocked |

---

## Repository contents

### `toy_sandwich_detector_quadratic.py`

Core simulation module. One call to `run_single()` simulates one configuration.

#### Geometry

```
Su  -->  u1 --> u2 --> ... --> uN
                                     D1  D2  ...  DN   (detector strip)
Sl  -->  l1 --> l2 --> ... --> lN
```

Upper row: `u_i` feeds detector `D_i` (same column, straight through).

Lower row: `l_i` feeds detector `D_j` via a quadratic mapping:

```
x       = col - center
q       = int(round(quad_alpha * x^2))
det_col = ((col - 1 - q) % n_col) + 1
```

Center columns map near-diagonally (small `q`); edge columns are deflected
further. At `quad_alpha = 0`, the lower-row routing is identical to the upper-row routing.

#### Key parameters

| Parameter | Default | Meaning |
|---|---|---|
| `m` | 64 | Phase cycle length |
| `tau0` | 32 | Internal buffer delay (`m // 2`) |
| `n_ticks` | 960 | Simulation duration |
| `n_columns` | 32 | Number of transport columns and detectors |
| `capacity_detector` | 1 | Maximum signals accepted per detector per tick |
| `source_period` | 8 | Both sources fire every N ticks |
| `quad_alpha` | 1/48 | Quadratic curvature of lower-row routing |
| `quad_center` | 16.5 | Center column for the quadratic map |

#### Sweep variable: `delta_phi`

The tick offset between the upper and lower source fire times.

- `delta_phi = 0`: both sources fire in phase. At columns where the lower-row
  routing still overlaps strongly with the upper row, tokens arrive at the same
  detector on the same tick. Capacity blocks one of them, reducing throughput.
- `delta_phi != 0`: tokens arrive on different ticks more often, so both are
  more likely to pass. Total throughput increases.

#### Suppression ratio

```
S(alpha) = N(delta_phi=0) / N(delta_phi=phi_max)
```

`N` is the total detector count summed over all columns, and `phi_max` is the
tested phase offset that gives the highest throughput for a given geometry.

- `S(alpha) ≈ 0.5` — strong suppression
- `S(alpha) ≈ 1.0` — weak or no suppression within the tested phase set

#### Exported functions

| Function | Purpose |
|---|---|
| `run_single(delta_phi, params)` | Simulate one episode |
| `lower_detector_col_for_sender(col, params)` | Quadratic routing for one column |
| `build_lower_detector_map(params)` | Full routing map as a list |
| `phi0_for_column(column, t0, tau0, m)` | Resonant initial phase for one column |

---

### `run_paper_sandwich_suite.py`

Paper experiment runner. Imports from `toy_sandwich_detector_quadratic.py` and
runs a fixed matrix of three geometries over selected phase offsets.

#### Experiment matrix

| Geometry | `quad_alpha` | Description |
|---|---|---|
| `flat` | 0.0 | Identity routing; each lower sender maps to the same detector column as the upper sender |
| `quadratic_weak` | 1/96 | Gentle bow; partial redistribution |
| `quadratic_strong` | 1/48 | Stronger bow; further redistribution |

Phase offsets for the paper suite: `[0, 16, 31, 32, 33, 48]`

#### Full sweep (flat geometry)

Before the paper suite, the script runs a complete sweep of `delta_phi ∈ {0,…,63}`
for the flat geometry (`quad_alpha = 0`). This establishes `phi_max` rigorously
from the full phase space rather than from the sparse list above.

Result: `phi_max = 1`, `N_max = 450`, `N(delta_phi=0) = 225`, `S = 0.500`
exactly. The suppression is precisely 2:1 in the flat case.

Outputs saved under `Output/paper_sandwich/flat_full_sweep/`:

| File | Content |
|---|---|
| `total_vs_phase_full.png` | Bar chart of total throughput for all 64 phase offsets |
| `full_sweep.json` | `phi_min`, `phi_max`, totals for all `delta_phi` |
| `full_sweep.npz` | Arrays `delta_phi` and `totals` |

#### Outputs per geometry

Saved under `Output/paper_sandwich/<geometry>/`:

| File | Content |
|---|---|
| `geometry_map.png` | Lower-row sender → detector column mapping |
| `heatmap.png` | Detector counts vs `delta_phi` |
| `selected_profiles.png` | Detector profiles for each tested phase offset |
| `total_vs_phase.png` | Total throughput vs phase offset |
| `suppression_profile.png` | Per-column: `profile(delta=0) − profile(phi_max)` |
| `timeline_delta0.png` | Raster + cumulative hits at `delta_phi = 0` |
| `timeline_delta_phi_min.png` | Raster + cumulative hits at minimum-throughput offset |
| `timeline_delta32.png` | Raster + cumulative hits at `delta_phi = 32` |

#### Aggregate outputs

Saved under `Output/paper_sandwich/`:

| File | Content |
|---|---|
| `summary.json` | All geometry stats, run data, and S(alpha) table |
| `summary.npz` | Profile matrices, totals, routing maps, S(alpha) arrays |
| `s_alpha_vs_alpha.png` | S(alpha) vs alpha line plot |
| `s_alpha_by_geometry.png` | S(alpha) bar chart by geometry |
| `s_alpha_labels.json` | Geometry labels for the saved arrays |

#### Console summary

```
geometry                   alpha  phi_max      N0    Nmax   S(alpha)
flat                     0.00000        1     225     450      0.500
quadratic_weak           0.01042        1     352     450      0.782
quadratic_strong         0.02083        1     380     450      0.844
```

`phi_max = 1` is confirmed by the full sweep. S(alpha) increases with geometry
strength: stronger quadratic curvature redistributes lower-row tokens across
columns, reduces coincident arrivals, and weakens global suppression.

---

## Running

Requirements: Python 3.9+, `numpy`, `matplotlib`.

```bash
# Full paper suite (three geometries, all plots and data)
python run_paper_sandwich_suite.py

# Single-geometry explorer
python toy_sandwich_detector_quadratic.py
```

Both scripts write outputs under an `Output/` subdirectory alongside the code.
No external data files are required.

---

## Main result

In the flat geometry, suppression at `delta_phi = 0` is strongest:

- Both sources fire in phase.
- Every lower-row token lands at the same detector column as the corresponding
  upper-row token.
- Tokens frequently reach the same detector on the same tick.
- The detector capacity limit blocks one of them.
- Total throughput drops to exactly half the maximum value (`S = 0.500`,
  confirmed by a full sweep over Δφ ∈ {0,…,63}).

Introducing quadratic curvature redistributes the lower-row routing. Near the
centre, arrivals still overlap strongly and suppression remains significant.
Toward the edges, the lower-row signals are displaced, conflicts are reduced,
and throughput recovers. The result is a geometry-tunable, position-dependent
suppression profile produced entirely by discrete timing and capacity constraints.

---

## Relation to DGO

This repository implements a deliberately minimal toy model related to the broader
DGO framework. It does not implement the full DGO substrate or co-evolving
geometry. Instead, it isolates one narrow question:

> Can interference-like suppression arise from discrete relational timing and
> finite capacity alone?

The present code should therefore be read as a mechanism study, not as a complete
realisation of the full framework.
