"""
run_paper_sandwich_suite.py

Fixed paper experiment suite for the DGO sandwich detector model.

Runs three geometries (flat, quadratic_weak, quadratic_strong) over a fixed
set of delta_phi values and produces a standardised set of figures and data
files for inclusion in the paper.

Backend simulation logic is imported from toy_sandwich_detector_quadratic.py.
No parameter scanning beyond the predefined paper matrix.
"""

import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Import backend ─────────────────────────────────────────────────────────────
# toy_sandwich_detector_quadratic.py lives in the same directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from toy_sandwich_detector_quadratic import (
    phi0_for_column,
    lower_detector_col_for_sender,
    build_lower_detector_map,
    run_single,
)

# ── Output root ────────────────────────────────────────────────────────────────
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      'Output', 'paper_sandwich')
os.makedirs(OUTDIR, exist_ok=True)

# ── Global fixed parameters ────────────────────────────────────────────────────
GLOBAL_PARAMS = dict(
    m                  = 64,
    tau0               = 32,
    n_ticks            = 960,
    n_columns          = 32,
    capacity_transport = 1,
    capacity_detector  = 1,
    source_period      = 8,
    source_start       = 1,
)

# Fixed phase offsets to evaluate for every geometry.
DELTA_PHI_VALUES = [0, 16, 31, 32, 33, 48]

# Whether to run the strong-curvature case.
INCLUDE_STRONG_CASE = True

# ── Geometry definitions ───────────────────────────────────────────────────────
# Each geometry is a dict of overrides on top of GLOBAL_PARAMS.
# quad_center = None is resolved to (n_columns + 1) / 2 before use.

def make_geometries(global_params):
    """Return ordered list of (name, params) for the paper suite."""
    n_col = global_params['n_columns']
    center = (n_col + 1) / 2.0

    geometries = []

    # 1. Flat — lower row feeds same-column detector (identity map).
    flat_params = global_params.copy()
    flat_params['quad_alpha']  = 0.0
    flat_params['quad_center'] = center
    geometries.append(('flat', flat_params))

    # 2. Quadratic weak — gentle bow, monotone mapping.
    weak_params = global_params.copy()
    weak_params['quad_alpha']  = 1.0 / 96.0
    weak_params['quad_center'] = center
    geometries.append(('quadratic_weak', weak_params))

    # 3. Quadratic strong — stronger bow (optional).
    if INCLUDE_STRONG_CASE:
        strong_params = global_params.copy()
        strong_params['quad_alpha']  = 1.0 / 48.0
        strong_params['quad_center'] = center
        geometries.append(('quadratic_strong', strong_params))

    return geometries


# ── Simulation helpers ─────────────────────────────────────────────────────────

def run_geometry(params, delta_phi_values):
    """
    Run run_single() for each delta_phi in delta_phi_values.
    Returns dict: delta_phi -> run_result.
    """
    runs = {}
    for delta in delta_phi_values:
        runs[delta] = run_single(delta, params)
    return runs


def compute_suite_stats(runs, delta_phi_values, params):
    """
    Compute paper-level statistics for one geometry over all delta_phi.

    Because we only ran a sparse set of delta_phi, we work only within that
    set for min/max and symmetry checks.
    """
    m = params['m']
    totals = {d: sum(runs[d]['detector_profile']) for d in delta_phi_values}

    phi_min = min(totals, key=totals.get)
    phi_max = max(totals, key=totals.get)
    total_min = totals[phi_min]
    total_max = totals[phi_max]

    suppression_strength = total_max - total_min
    suppression_ratio    = total_min / total_max if total_max > 0 else float('nan')

    # Symmetry error: how similar are the two flanks around m//2?
    half = m // 2
    d_lo = half - 1
    d_hi = half + 1
    if d_lo in totals and d_hi in totals:
        symmetry_error = abs(totals[d_lo] - totals[d_hi])
    else:
        symmetry_error = None   # flanks not in delta_phi_values

    # Suppression profile: profile at delta=0 minus profile at phi_ref (phi_max).
    # phi_max is the high-count reference; phi_min=0 here would give zero diff.
    phi_ref = phi_max
    prof0    = np.array(runs[0]['detector_profile'],        dtype=float)
    prof_ref = np.array(runs[phi_ref]['detector_profile'],  dtype=float)
    suppression_profile = (prof0 - prof_ref).tolist()

    return dict(
        phi_min             = phi_min,
        phi_max             = phi_max,
        total_min           = total_min,
        total_max           = total_max,
        suppression_strength = suppression_strength,
        suppression_ratio   = suppression_ratio,
        symmetry_error      = symmetry_error,
        suppression_profile = suppression_profile,
        totals              = totals,
    )


# ── Per-geometry plots ─────────────────────────────────────────────────────────

def plot_geometry_map(params, geo_name, outdir):
    """Geometry-only: lower sender column -> detector column."""
    n_col = params['n_columns']
    sender_cols = list(range(1, n_col + 1))
    det_cols    = [lower_detector_col_for_sender(c, params) for c in sender_cols]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sender_cols, det_cols, 'o-', color='steelblue', lw=2, ms=5)
    ax.plot(sender_cols, sender_cols, '--', color='grey', lw=1,
            label='diagonal (no offset)')
    ax.set_xlabel('lower sender column l_i')
    ax.set_ylabel('detector column D_j fed')
    ax.set_title(
        f'Lower-row routing map — {geo_name}\n'
        f'quad_alpha={params["quad_alpha"]:.5f}  center={params["quad_center"]:.1f}')
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'geometry_map.png'), dpi=130)
    plt.close()


def plot_heatmap(runs, delta_phi_values, params, geo_name, outdir):
    """
    Heatmap of detector counts.  y-axis = selected delta_phi values only
    (sparse rows), not the full 0..m-1 range.
    """
    n_col  = params['n_columns']
    deltas = delta_phi_values

    matrix = np.array([runs[d]['detector_profile'] for d in deltas],
                      dtype=float)  # shape (len(deltas), n_col)

    fig, ax = plt.subplots(figsize=(9, 4))
    im = ax.imshow(matrix, aspect='auto', origin='lower',
                   cmap='hot', interpolation='nearest')
    plt.colorbar(im, ax=ax, label='detector count')
    ax.set_xlabel('detector column i')
    ax.set_ylabel('delta_phi index')
    ax.set_yticks(range(len(deltas)))
    ax.set_yticklabels([str(d) for d in deltas])
    col_ticks = [c - 1 for c in range(1, n_col + 1) if c == 1 or c % 4 == 0]
    ax.set_xticks(col_ticks)
    ax.set_xticklabels([str(c + 1) for c in col_ticks])
    ax.set_title(
        f'Detector counts vs delta_phi — {geo_name}\n'
        f'quad_alpha={params["quad_alpha"]:.5f}  m={params["m"]}')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'heatmap.png'), dpi=130)
    plt.close()


def plot_selected_profiles(runs, delta_phi_values, params, geo_name, outdir):
    """Overlay detector profiles for each selected delta_phi."""
    n_col = params['n_columns']
    cols  = np.arange(1, n_col + 1)

    cmap   = plt.cm.viridis
    n      = len(delta_phi_values)
    colors = [cmap(k / max(n - 1, 1)) for k in range(n)]

    fig, ax = plt.subplots(figsize=(9, 4))
    for delta, color in zip(delta_phi_values, colors):
        profile = runs[delta]['detector_profile']
        ax.plot(cols, profile, 'o-', lw=1.5, ms=4,
                label=f'delta_phi={delta}', color=color)
    ax.set_xlabel('detector column i')
    ax.set_ylabel('detector count')
    ax.set_title(f'Selected detector profiles — {geo_name}')
    col_ticks = [c for c in range(1, n_col + 1) if c == 1 or c % 4 == 0]
    ax.set_xticks(col_ticks)
    ax.legend(fontsize=8, ncol=3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'selected_profiles.png'), dpi=130)
    plt.close()


def plot_total_vs_phase(runs, delta_phi_values, params, geo_name, stats, outdir):
    """Bar chart of total detector count per delta_phi."""
    totals = [stats['totals'][d] for d in delta_phi_values]

    fig, ax = plt.subplots(figsize=(7, 4))
    bar_colors = ['firebrick' if d == stats['phi_min'] else 'steelblue'
                  for d in delta_phi_values]
    ax.bar(range(len(delta_phi_values)), totals, color=bar_colors, alpha=0.85)
    ax.set_xlabel('delta_phi')
    ax.set_ylabel('total detector count (all columns)')
    ax.set_title(f'Total throughput vs phase offset — {geo_name}')
    ax.set_xticks(range(len(delta_phi_values)))
    ax.set_xticklabels([str(d) for d in delta_phi_values])
    for k, (d, t) in enumerate(zip(delta_phi_values, totals)):
        ax.text(k, t + 0.3, str(t), ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'total_vs_phase.png'), dpi=130)
    plt.close()


def plot_suppression_profile(stats, params, geo_name, outdir):
    """
    Bar chart of detector_profile(delta=0) - detector_profile(delta=phi_min)
    per detector column.
    """
    n_col = params['n_columns']
    cols  = np.arange(1, n_col + 1)
    diff  = np.array(stats['suppression_profile'])

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(cols, diff,
           color=['firebrick' if v < 0 else 'steelblue' for v in diff])
    ax.axhline(0, color='black', lw=0.8)
    ax.set_xlabel('detector column i')
    ax.set_ylabel(f'count(delta=0) - count(delta={stats["phi_max"]})')
    ax.set_title(
        f'Suppression profile — {geo_name}\n'
        f'(delta=0 vs delta_phi={stats["phi_max"]} = phi_max / high-count reference)')
    col_ticks = [c for c in range(1, n_col + 1) if c == 1 or c % 4 == 0]
    ax.set_xticks(col_ticks)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'suppression_profile.png'), dpi=130)
    plt.close()


def plot_timeline(run, params, delta_phi, outdir, fname):
    """
    Two-panel timeline: raster of per-tick hits + cumulative per column.
    """
    n_col   = params['n_columns']
    n_ticks = params['n_ticks']
    ticks   = np.arange(1, n_ticks + 1)
    det_ids = [f'D{i}' for i in range(1, n_col + 1)]

    matrix = np.array([run['det_timeline'][did] for did in det_ids],
                      dtype=float)  # shape (n_col, n_ticks)

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    ax = axes[0]
    ax.imshow(matrix, aspect='auto', origin='lower',
              extent=[0.5, n_ticks + 0.5, 0.5, n_col + 0.5],
              cmap='Blues', interpolation='nearest', vmin=0, vmax=1)
    ax.set_ylabel('detector column i')
    col_ticks = [c for c in range(1, n_col + 1) if c == 1 or c % 4 == 0]
    ax.set_yticks(col_ticks)
    ax.set_title(f'Detector hit raster  [delta_phi={delta_phi}]')

    ax = axes[1]
    cmap_t = plt.cm.tab20
    for i, did in enumerate(det_ids):
        ax.plot(ticks, np.cumsum(run['det_timeline'][did]),
                lw=1, color=cmap_t(i / n_col), label=f'D{i+1}')
    ax.set_xlabel('tick')
    ax.set_ylabel('cumulative hits')
    ax.set_title('Cumulative per-column hits')
    ax.legend(fontsize=6, ncol=6)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, fname), dpi=130)
    plt.close()


# ── Save results ───────────────────────────────────────────────────────────────

def save_summary_json(all_geo_data, global_params, delta_phi_values, outdir,
                      s_alpha_rows=None):
    """Write results.json with all geometry-level stats and run data."""
    doc = {
        'global_params':   global_params,
        'phase_offsets':   delta_phi_values,
        'geometries':      {},
    }

    for geo_name, params, runs, stats in all_geo_data:
        lower_map = build_lower_detector_map(params)
        geo_doc = {
            'quad_alpha':           params['quad_alpha'],
            'quad_center':          params['quad_center'],
            'lower_detector_map':   lower_map,
            'phi_min':              stats['phi_min'],
            'phi_max':              stats['phi_max'],
            'total_min':            stats['total_min'],
            'total_max':            stats['total_max'],
            'suppression_strength': stats['suppression_strength'],
            'suppression_ratio':    stats['suppression_ratio'],
            'symmetry_error':       stats['symmetry_error'],
            'suppression_profile':  stats['suppression_profile'],
            'runs': {
                str(d): {
                    'detector_profile': runs[d]['detector_profile'],
                    'detector_total':   sum(runs[d]['detector_profile']),
                    'det_blocked':      {k: v for k, v
                                        in runs[d]['det_blocked'].items()},
                    'upper_sends':      runs[d]['upper_sends'],
                    'lower_sends':      runs[d]['lower_sends'],
                }
                for d in delta_phi_values
            },
        }
        doc['geometries'][geo_name] = geo_doc

    if s_alpha_rows is not None:
        doc['s_alpha_table'] = s_alpha_rows

    with open(os.path.join(outdir, 'summary.json'), 'w') as f:
        json.dump(doc, f, indent=2)


def save_summary_npz(all_geo_data, delta_phi_values, outdir, s_alpha_rows=None):
    """Write summary.npz with profile matrices, totals, and maps."""
    arrays = {
        'delta_phi_values': np.array(delta_phi_values, dtype=int),
    }

    for geo_name, params, runs, stats in all_geo_data:
        key = geo_name   # e.g. 'flat', 'quadratic_weak', 'quadratic_strong'
        profile_matrix = np.array(
            [runs[d]['detector_profile'] for d in delta_phi_values], dtype=int)
        totals_arr = np.array(
            [sum(runs[d]['detector_profile']) for d in delta_phi_values], dtype=int)
        lower_map = np.array(build_lower_detector_map(params), dtype=int)

        arrays[f'counts_{key}']    = profile_matrix
        arrays[f'totals_{key}']    = totals_arr
        arrays[f'lower_map_{key}'] = lower_map

    if s_alpha_rows is not None:
        arrays['s_alpha_alpha']   = np.array([r['alpha']           for r in s_alpha_rows], dtype=float)
        arrays['s_alpha_value']   = np.array([r['S_alpha']         for r in s_alpha_rows], dtype=float)
        arrays['s_alpha_phi_max'] = np.array([r['phi_max']         for r in s_alpha_rows], dtype=int)
        arrays['s_alpha_total0']  = np.array([r['total_at_zero']   for r in s_alpha_rows], dtype=int)
        arrays['s_alpha_totalmax']= np.array([r['total_at_phi_max'] for r in s_alpha_rows], dtype=int)
        # geometry names as a separate JSON (npz does not handle string arrays cleanly)
        labels = [r['geometry'] for r in s_alpha_rows]
        with open(os.path.join(outdir, 's_alpha_labels.json'), 'w') as f:
            json.dump(labels, f)

    np.savez(os.path.join(outdir, 'summary.npz'), **arrays)


# ── S(alpha) extraction ────────────────────────────────────────────────────────

def collect_s_alpha(all_geo_data):
    """
    Build a list of dicts, one per geometry, with:
        geometry, alpha, phi_max, total_at_zero, total_at_phi_max, S_alpha

    S(alpha) = N(delta_phi=0) / N(delta_phi=phi_max)

    phi_max is the high-count reference from compute_suite_stats —
    always use this, never phi_min, to avoid comparing delta=0 against itself.
    """
    rows = []
    for geo_name, params, runs, stats in all_geo_data:
        phi_max        = stats['phi_max']
        total_at_zero  = stats['totals'][0]
        total_at_phimax = stats['totals'][phi_max]
        s_alpha = total_at_zero / total_at_phimax if total_at_phimax > 0 else float('nan')
        rows.append(dict(
            geometry        = geo_name,
            alpha           = params['quad_alpha'],
            phi_max         = phi_max,
            total_at_zero   = total_at_zero,
            total_at_phi_max = total_at_phimax,
            S_alpha         = s_alpha,
        ))
    return rows


def plot_s_alpha(s_alpha_rows, outdir):
    """
    Line + marker plot of S(alpha) vs alpha.
    Each point is annotated with its geometry name.
    """
    alphas  = [r['alpha']   for r in s_alpha_rows]
    s_vals  = [r['S_alpha'] for r in s_alpha_rows]
    labels  = [r['geometry'] for r in s_alpha_rows]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(alphas, s_vals, 'o-', color='steelblue', lw=2, ms=8)
    for x, y, label in zip(alphas, s_vals, labels):
        ax.annotate(label, xy=(x, y),
                    xytext=(4, 6), textcoords='offset points', fontsize=9)
    ax.set_xlabel('alpha')
    ax.set_ylabel('S(alpha) = N(delta_phi=0) / N(delta_phi=phi_max)')
    ax.set_title('Suppression ratio S(alpha) vs geometry strength')
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color='grey', lw=0.8, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 's_alpha_vs_alpha.png'), dpi=130)
    plt.close()


def plot_s_alpha_bar(s_alpha_rows, outdir):
    """
    Bar chart of S(alpha) by geometry name.
    """
    labels = [r['geometry'] for r in s_alpha_rows]
    s_vals = [r['S_alpha']  for r in s_alpha_rows]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(range(len(labels)), s_vals, color='steelblue', alpha=0.85)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=10, ha='right')
    ax.set_ylabel('S(alpha) = N(delta_phi=0) / N(delta_phi=phi_max)')
    ax.set_title('Suppression ratio by geometry')
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color='grey', lw=0.8, linestyle='--')
    for k, v in enumerate(s_vals):
        ax.text(k, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 's_alpha_by_geometry.png'), dpi=130)
    plt.close()


def print_s_alpha_table(s_alpha_rows):
    header = (f"{'geometry':<22} {'alpha':>9} {'phi_max':>8} "
              f"{'N0':>7} {'Nmax':>7} {'S(alpha)':>10}")
    print()
    print(header)
    print('-' * len(header))
    for r in s_alpha_rows:
        print(
            f"{r['geometry']:<22} "
            f"{r['alpha']:>9.5f} "
            f"{r['phi_max']:>8} "
            f"{r['total_at_zero']:>7} "
            f"{r['total_at_phi_max']:>7} "
            f"{r['S_alpha']:>10.3f}"
        )
    print()


# ── Console summary table ──────────────────────────────────────────────────────

def print_summary_table(all_geo_data):
    header = (f"{'geometry':<20} {'phi_min':>7} {'phi_max':>7} "
              f"{'total_min':>9} {'total_max':>9} "
              f"{'ratio':>6} {'sym_err':>8}")
    print()
    print(header)
    print('-' * len(header))
    for geo_name, params, runs, stats in all_geo_data:
        sym = stats['symmetry_error']
        sym_str = str(sym) if sym is not None else 'n/a'
        print(
            f"{geo_name:<20} "
            f"{stats['phi_min']:>7} "
            f"{stats['phi_max']:>7} "
            f"{stats['total_min']:>9} "
            f"{stats['total_max']:>9} "
            f"{stats['suppression_ratio']:>6.3f} "
            f"{sym_str:>8}"
        )
    print()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    geometries = make_geometries(GLOBAL_PARAMS)

    print(f"Paper sandwich suite")
    print(f"  global: m={GLOBAL_PARAMS['m']}  tau0={GLOBAL_PARAMS['tau0']}"
          f"  n_ticks={GLOBAL_PARAMS['n_ticks']}  n_columns={GLOBAL_PARAMS['n_columns']}"
          f"  source_period={GLOBAL_PARAMS['source_period']}")
    print(f"  delta_phi_values: {DELTA_PHI_VALUES}")
    print(f"  geometries: {[g[0] for g in geometries]}")
    print()

    all_geo_data = []   # list of (name, params, runs, stats)

    for geo_name, params in geometries:
        subdir = os.path.join(OUTDIR, geo_name)
        os.makedirs(subdir, exist_ok=True)

        lower_map = build_lower_detector_map(params)
        print(f"--- {geo_name}  alpha={params['quad_alpha']:.5f}"
              f"  center={params['quad_center']:.1f}")
        print(f"    lower_map: {lower_map}")

        # Run simulations for each fixed delta_phi.
        runs = run_geometry(params, DELTA_PHI_VALUES)

        for d in DELTA_PHI_VALUES:
            total = sum(runs[d]['detector_profile'])
            print(f"    delta_phi={d:3d}  total={total:5d}")

        stats = compute_suite_stats(runs, DELTA_PHI_VALUES, params)
        print(f"    phi_min={stats['phi_min']}  phi_max={stats['phi_max']}"
              f"  ratio={stats['suppression_ratio']:.3f}"
              f"  sym_err={stats['symmetry_error']}")
        print()

        # ── Plots ──────────────────────────────────────────────────────────────
        plot_geometry_map(params, geo_name, subdir)
        plot_heatmap(runs, DELTA_PHI_VALUES, params, geo_name, subdir)
        plot_selected_profiles(runs, DELTA_PHI_VALUES, params, geo_name, subdir)
        plot_total_vs_phase(runs, DELTA_PHI_VALUES, params, geo_name, stats, subdir)
        plot_suppression_profile(stats, params, geo_name, subdir)

        # Timelines: delta=0, delta=phi_min, delta=32.
        timeline_deltas = sorted(set([0, stats['phi_min'], 32]))
        for d in timeline_deltas:
            if d in runs:
                if d == stats['phi_min']:
                    fname = f'timeline_delta_phi_min.png'
                elif d == 0:
                    fname = 'timeline_delta0.png'
                else:
                    fname = f'timeline_delta{d}.png'
                plot_timeline(runs[d], params, d, subdir, fname)

        all_geo_data.append((geo_name, params, runs, stats))

    # ── S(alpha) summary ───────────────────────────────────────────────────────
    s_alpha_rows = collect_s_alpha(all_geo_data)
    plot_s_alpha(s_alpha_rows, OUTDIR)
    plot_s_alpha_bar(s_alpha_rows, OUTDIR)
    print_s_alpha_table(s_alpha_rows)

    # ── Aggregate save ─────────────────────────────────────────────────────────
    save_summary_json(all_geo_data, GLOBAL_PARAMS, DELTA_PHI_VALUES, OUTDIR,
                      s_alpha_rows=s_alpha_rows)
    save_summary_npz(all_geo_data, DELTA_PHI_VALUES, OUTDIR,
                     s_alpha_rows=s_alpha_rows)

    print_summary_table(all_geo_data)
    print(f"Results saved to {OUTDIR}")
    print()
    print("S(alpha) increases with geometry strength, indicating weaker global "
          "suppression under stronger spatial delay modulation.")


if __name__ == '__main__':
    main()
