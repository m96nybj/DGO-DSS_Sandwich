"""
toy_sandwich_detector_quadratic.py

Sandwich detector toy model — DGO-style processor-node timing,
quadratic lower-row routing geometry.

Two transport rows (upper u1..uN and lower l1..lN) feed a strip of
detector nodes D1..DN.  Upper row feeds same-column detectors.
Lower row feeds detectors via a quadratic position-dependent offset:

    x = col - center
    q = int(round(quad_alpha * x * x))
    det_col = ((col - 1 - q) % n_col) + 1

Purpose: test whether wave-like fringe structure (bowed/curved bright
and dark bands in the detector heatmap) can emerge from discrete timing
and local capacity alone — without complex amplitudes, wave equations, or
phase memory.

  delta_phi = 0 : upper and lower tokens arrive at Di on the same tick
                  at columns near the center (small q), but with a delay
                  mismatch at edge columns (large q).
                  -> capacity_detector = 1 suppresses coincident arrivals.
                  -> suppression depth is position-dependent.

  delta_phi != 0: tokens arrive on different ticks (most columns).
                  -> no capacity conflict -> both pass.
                  -> the crossover between suppress and no-suppress
                     shifts with both delta_phi and detector column.

Ontology:
  signals            — unit events; carry only a row-provenance tag
  recv_open          — node accepts a signal when local theta % m == 0
  send_open          — node forwards a signal when local theta % m == tau0
  buffer             — signals sit for tau0 ticks between recv and send
  capacity_transport — max signals accepted per transport node per tick
  capacity_detector  — max signals accepted per detector per tick (gating)
  detector Di        — counts accepted signals; no forwarding, no phase state
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Output directory ───────────────────────────────────────────────────────────
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      'Output', 'sandwich-1-48-dphi31')
os.makedirs(OUTDIR, exist_ok=True)

# ── Default parameters ─────────────────────────────────────────────────────────
DEFAULT_PARAMS = dict(
    m                  = 64,          # phase cycle length
    tau0               = 32,          # half-turn internal delay (m // 2)
    n_ticks            = 960,         # simulation length
    n_columns          = 32,          # number of columns (transport + detector)
    capacity_transport = 1,           # max signals accepted per transport node
    capacity_detector  = 1,           # max signals accepted per detector per tick
    source_period      = 8,           # both sources fire every source_period ticks
    source_start       = 1,           # tick of first upper-source fire
    geometry_mode      = "quadratic",
    quad_alpha         = 1 / 48,  # curvature; increase for stronger bowing
    quad_center        = None,        # None -> auto: (n_columns + 1) / 2
    # Selected delta_phi values to plot individually (built in main() from m).
    selected_delta_phi          = 31,   # None -> auto
    selected_timeline_delta_phi = 31,   # None -> auto
    save_full_heatmap      = True,
    save_selected_profiles = True,
    save_total_vs_phase    = True,
    save_timeline          = True,
)

# ── Phase helpers ──────────────────────────────────────────────────────────────

def phi0_for_column(column, t0_source, tau0, m):
    """
    Initial theta for a transport node at column i (1-indexed) so that it is
    recv_open exactly one tick after the signal from the source arrives.

    Tick order each turn: RECEIVE -> SEND -> SOURCE -> PHASE_UPDATE.
    Source fires at tick t0_source; signal lands in new_arriving, processed in
    RECEIVE of tick t0_source+1.  Each subsequent hop: recv at T,
    send at T+tau0, next recv at T+tau0+1.  Column i receives at:
        t_recv(i) = t0_source + 1 + (i-1) * (tau0 + 1)

    recv_open at tick T iff (phi0 + T - 1) % m == 0
    -> phi0 = (-(t0_source + (i-1)*(tau0+1))) % m
    """
    return (-(t0_source + (column - 1) * (tau0 + 1))) % m


# ── Quadratic geometry ─────────────────────────────────────────────────────────

def lower_detector_col_for_sender(col, params):
    """
    Given lower-row sender column col (1-indexed), compute which detector
    column this lower signal feeds under quadratic geometry.

    Let x = col - center.
    Define integer quadratic offset:
        q = int(round(quad_alpha * x * x))

    Then feed detector:
        det_col = ((col - 1 - q) % n_col) + 1

    Center columns (x ~ 0) have small q -> map near-diagonally.
    Edge columns (|x| large) have larger q -> map further away.
    """
    n_col      = params['n_columns']
    quad_alpha = params['quad_alpha']
    center     = params['quad_center']   # already resolved to a float in main()
    x = col - center
    q = int(round(quad_alpha * x * x))
    det_col = ((col - 1 - q) % n_col) + 1
    return det_col


def build_lower_detector_map(params):
    """Return list of length n_col: lower_detector_map[j-1] = det_col for sender j."""
    n_col = params['n_columns']
    return [lower_detector_col_for_sender(j, params) for j in range(1, n_col + 1)]


# ── Single simulation run ──────────────────────────────────────────────────────

def run_single(delta_phi, params):
    """
    Simulate one episode with lower source offset by delta_phi ticks.

    Returns a diagnostics dict including:
      detector_profile : list[n_columns]  — total counts at each Di
      det_timeline     : dict[Di -> list[int]]  — per-tick accepted count
      det_blocked      : dict[Di -> int]   — total blocked signals at each Di
      upper_sends      : list[int]         — total times each u_i sent
      lower_sends      : list[int]         — total times each l_i sent
    """
    m          = params['m']
    tau0       = params['tau0']
    n_ticks    = params['n_ticks']
    n_col      = params['n_columns']
    cap_t      = params['capacity_transport']
    cap_d      = params['capacity_detector']
    src_period = params['source_period']
    src_start  = params['source_start']

    t0_upper = src_start
    t0_lower = src_start + delta_phi   # lower source fires delta_phi ticks later

    # Node IDs
    upper_ids     = [f'u{i}' for i in range(1, n_col + 1)]
    lower_ids     = [f'l{i}' for i in range(1, n_col + 1)]
    det_ids       = [f'D{i}' for i in range(1, n_col + 1)]
    transport_ids = upper_ids + lower_ids

    # Initial phases: each transport node tuned so it is recv_open exactly
    # when the first token from its source arrives.
    theta = {}
    for i in range(1, n_col + 1):
        theta[f'u{i}'] = phi0_for_column(i, t0_upper, tau0, m)
        theta[f'l{i}'] = phi0_for_column(i, t0_lower, tau0, m)
    # Detectors have no phase gating — acceptance is capacity-only.

    # Buffers: each transport node holds signals waiting tau0 ticks.
    buffer = {nid: [] for nid in transport_ids}

    # arriving[nid]: signals delivered to nid at the start of this tick.
    arriving = {nid: [] for nid in transport_ids + det_ids}

    # Diagnostics
    det_counts   = {did: 0 for did in det_ids}
    det_blocked  = {did: 0 for did in det_ids}
    det_timeline = {did: [] for did in det_ids}
    upper_sends  = [0] * n_col
    lower_sends  = [0] * n_col

    for t in range(1, n_ticks + 1):

        # Signals routed during this tick (populate arriving for next tick).
        new_arriving = {nid: [] for nid in transport_ids + det_ids}

        # Per-tick accepted counter (enforces capacity).
        tick_accepted = {nid: 0 for nid in transport_ids + det_ids}

        # ── Step 1: RECEIVE ───────────────────────────────────────────────────
        # Detectors: no phase gating; accept up to cap_d signals per tick.
        for did in det_ids:
            n_hit = 0
            for sig in arriving[did]:
                if tick_accepted[did] < cap_d:
                    det_counts[did]   += 1
                    tick_accepted[did] += 1
                    n_hit += 1
                else:
                    det_blocked[did] += 1
            det_timeline[did].append(n_hit)

        # Transport nodes: accept if recv_open (theta % m == 0) and capacity.
        for nid in transport_ids:
            if theta[nid] % m == 0:               # recv_open
                for sig in arriving[nid]:
                    if tick_accepted[nid] < cap_t:
                        buffer[nid].append({'ready_at': t + tau0, 'from': sig['from']})
                        tick_accepted[nid] += 1
                    # else: token lost (blocked by transport capacity)

        # ── Step 2: SEND & BROADCAST ──────────────────────────────────────────
        # Transport nodes forward signals when send_open (theta % m == tau0).
        # Upper row: l[i] -> D[i]  (same column, no shift)
        # Lower row: l[i] -> D[lower_detector_col_for_sender(i, params)]
        # Both rows forward rightward to the next transport node.
        for row_ids, row_tag, send_list, is_lower in [
                (upper_ids, 'upper', upper_sends, False),
                (lower_ids, 'lower', lower_sends, True)]:
            for idx, nid in enumerate(row_ids):
                if theta[nid] % m == tau0:         # send_open
                    ready = [s for s in buffer[nid] if s['ready_at'] == t]
                    buffer[nid] = [s for s in buffer[nid] if s['ready_at'] != t]
                    for sig in ready:
                        col = idx + 1              # 1-indexed column number
                        # Quadratic lower-row routing: each sender feeds a
                        # position-dependent detector column.
                        if is_lower:
                            det_col = lower_detector_col_for_sender(col, params)
                        else:
                            det_col = col          # upper: straight through
                        new_arriving[f'D{det_col}'].append({'from': row_tag, 'col': col})
                        # Forward rightward to next transport node.
                        if idx + 1 < n_col:
                            new_arriving[row_ids[idx + 1]].append(
                                {'from': sig['from'], 'col': col})
                        send_list[idx] += 1

        # ── Step 3: SOURCE ────────────────────────────────────────────────────
        if t >= t0_upper and (t - t0_upper) % src_period == 0:
            new_arriving[upper_ids[0]].append({'from': 'upper', 'col': 0})

        if t >= t0_lower and (t - t0_lower) % src_period == 0:
            new_arriving[lower_ids[0]].append({'from': 'lower', 'col': 0})

        # ── Step 4: PHASE UPDATE ──────────────────────────────────────────────
        for nid in transport_ids:
            theta[nid] = (theta[nid] + 1) % m

        arriving = new_arriving

    detector_profile = [det_counts[f'D{i}'] for i in range(1, n_col + 1)]

    return dict(
        detector_profile = detector_profile,
        det_timeline     = {did: det_timeline[did] for did in det_ids},
        det_blocked      = {did: det_blocked[did]  for did in det_ids},
        upper_sends      = upper_sends,
        lower_sends      = lower_sends,
        delta_phi        = delta_phi,
    )


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_heatmap(all_runs, params, outdir):
    """
    Heatmap: x = detector column (1..n_col), y = delta_phi (0..m-1).
    Colour = detector count.  With quadratic routing the dark/bright bands
    should bow rather than translate linearly with delta_phi.
    """
    m     = params['m']
    n_col = params['n_columns']
    deltas = list(range(m))

    matrix = np.array([all_runs[d]['detector_profile'] for d in deltas],
                      dtype=float)  # shape (m, n_col)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(matrix, aspect='auto', origin='lower',
                   extent=[0.5, n_col + 0.5, -0.5, m - 0.5],
                   cmap='hot')
    plt.colorbar(im, ax=ax, label='detector count')
    ax.set_xlabel('detector column i')
    ax.set_ylabel('delta_phi (lower-source offset, ticks)')
    ax.set_title(
        f'Sandwich detector (quadratic routing): counts vs phase offset\n'
        f'quad_alpha={params["quad_alpha"]:.4f}  '
        f'center={params["quad_center"]:.1f}  m={m}')
    # Only label every 4th column to avoid crowding
    col_ticks = [c for c in range(1, n_col + 1) if c == 1 or c % 4 == 0]
    ax.set_xticks(col_ticks)
    # Label every 4th delta_phi
    phi_ticks = [d for d in range(m) if d % max(1, m // 16) == 0]
    ax.set_yticks(phi_ticks)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'detector_heatmap_vs_phase.png'), dpi=130)
    plt.close()


def plot_selected_profiles(all_runs, params, outdir):
    """
    Overlay of detector_profile[i] for a small representative set of delta_phi.
    """
    n_col    = params['n_columns']
    selected = params['selected_delta_phi']
    cols     = np.arange(1, n_col + 1)

    cmap   = plt.cm.viridis
    colors = [cmap(k / max(len(selected) - 1, 1)) for k in range(len(selected))]

    fig, ax = plt.subplots(figsize=(9, 4))
    for delta, color in zip(selected, colors):
        profile = all_runs[delta]['detector_profile']
        ax.plot(cols, profile, 'o-', lw=1.5, ms=4,
                label=f'delta_phi={delta}', color=color)
    ax.set_xlabel('detector column i')
    ax.set_ylabel('detector count')
    ax.set_title('Detector profiles for selected phase offsets (quadratic routing)')
    col_ticks = [c for c in range(1, n_col + 1) if c == 1 or c % 4 == 0]
    ax.set_xticks(col_ticks)
    ax.legend(fontsize=8, ncol=3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'detector_profiles_selected_offsets.png'), dpi=130)
    plt.close()


def plot_total_vs_phase(all_runs, params, outdir):
    """
    Total detector count (summed over all columns) vs delta_phi.
    Bar chart; shows global throughput change with source phase offset.
    """
    m      = params['m']
    deltas = list(range(m))
    totals = [sum(all_runs[d]['detector_profile']) for d in deltas]

    fig, ax = plt.subplots(figsize=(max(8, m // 4), 4))
    ax.bar(deltas, totals, color='steelblue', alpha=0.85)
    ax.set_xlabel('delta_phi (lower-source offset, ticks)')
    ax.set_ylabel('total detector count (all columns)')
    ax.set_title('Total detector throughput vs source phase offset (quadratic routing)')
    # Annotate only selected offsets to avoid crowding
    for x in params['selected_delta_phi']:
        ax.text(x, totals[x] + 0.3, str(totals[x]),
                ha='center', va='bottom', fontsize=7, color='firebrick')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'detector_total_vs_phase.png'), dpi=130)
    plt.close()


def plot_timeline(run, params, delta_phi, outdir):
    """
    Per-tick detector hit timeline for all Di in one run.
    Two stacked panels: raster of hits + cumulative per column.
    """
    n_col   = params['n_columns']
    n_ticks = params['n_ticks']
    ticks   = np.arange(1, n_ticks + 1)
    det_ids = [f'D{i}' for i in range(1, n_col + 1)]

    matrix = np.array([run['det_timeline'][did] for did in det_ids],
                      dtype=float)  # shape (n_col, n_ticks)

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    # Top: raster (each row = one detector column, colour = hit or not)
    ax = axes[0]
    ax.imshow(matrix, aspect='auto', origin='lower',
              extent=[0.5, n_ticks + 0.5, 0.5, n_col + 0.5],
              cmap='Blues', interpolation='nearest', vmin=0, vmax=1)
    ax.set_ylabel('detector column i')
    col_ticks = [c for c in range(1, n_col + 1) if c == 1 or c % 4 == 0]
    ax.set_yticks(col_ticks)
    ax.set_title(f'Detector hit raster  [delta_phi={delta_phi}  quad_alpha={params["quad_alpha"]:.4f}]')

    # Bottom: cumulative totals per column
    ax = axes[1]
    cmap   = plt.cm.tab20
    for i, did in enumerate(det_ids):
        ax.plot(ticks, np.cumsum(run['det_timeline'][did]),
                lw=1, color=cmap(i / n_col), label=f'D{i+1}')
    ax.set_xlabel('tick')
    ax.set_ylabel('cumulative hits')
    ax.set_title('Cumulative per-column hits')
    ax.legend(fontsize=6, ncol=6)
    plt.tight_layout()
    fname = f'timeline_delta{delta_phi}.png'
    plt.savefig(os.path.join(outdir, fname), dpi=130)
    plt.close()


def plot_lower_detector_map(params, outdir):
    """
    Geometry plot: lower sender column -> detector column.
    Shows the quadratic routing independent of dynamics.
    """
    n_col = params['n_columns']
    sender_cols = list(range(1, n_col + 1))
    det_cols    = [lower_detector_col_for_sender(c, params) for c in sender_cols]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sender_cols, det_cols, 'o-', color='steelblue', lw=2, ms=6)
    ax.plot(sender_cols, sender_cols, '--', color='grey', lw=1, label='diagonal (no offset)')
    ax.set_xlabel('lower sender column l_i')
    ax.set_ylabel('detector column D_j fed by l_i')
    ax.set_title(
        f'Quadratic lower-row routing map\n'
        f'quad_alpha={params["quad_alpha"]:.4f}  center={params["quad_center"]:.1f}')
    ax.set_xticks(range(1, n_col + 1, max(1, n_col // 8)))
    ax.set_yticks(range(1, n_col + 1, max(1, n_col // 8)))
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'lower_detector_map.png'), dpi=130)
    plt.close()


def plot_suppression_profile(all_runs, params, outdir):
    """
    Suppression profile: detector_profile(delta=0) minus detector_profile(delta=m//2).
    Positive -> delta=0 is higher count; negative -> delta=m//2 is higher.
    Reveals which detector columns are most affected by phase coincidence.
    """
    m     = params['m']
    n_col = params['n_columns']
    cols  = np.arange(1, n_col + 1)

    prof0    = np.array(all_runs[0]['detector_profile'], dtype=float)
    prof_mid = np.array(all_runs[m // 2]['detector_profile'], dtype=float)
    diff     = prof0 - prof_mid

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(cols, diff, color=['firebrick' if v < 0 else 'steelblue' for v in diff])
    ax.axhline(0, color='black', lw=0.8)
    ax.set_xlabel('detector column i')
    ax.set_ylabel('count(delta=0) - count(delta=m//2)')
    ax.set_title(
        f'Suppression profile: delta_phi=0 vs delta_phi={m//2}\n'
        f'(negative = half-period offset gives more hits)')
    ax.set_xticks(range(1, n_col + 1, max(1, n_col // 8)))
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'suppression_profile.png'), dpi=130)
    plt.close()


# ── Save results ───────────────────────────────────────────────────────────────

def save_results(all_runs, params, outdir):
    """Save summary JSON and NPZ."""
    m     = params['m']
    n_col = params['n_columns']

    lower_detector_map = build_lower_detector_map(params)

    # Build profile matrix: shape (m, n_col)
    profile_matrix = np.array(
        [all_runs[d]['detector_profile'] for d in range(m)], dtype=int)

    np.savez(os.path.join(outdir, 'results.npz'),
             profile_matrix    = profile_matrix,
             delta_phi         = np.arange(m),
             lower_detector_map = np.array(lower_detector_map, dtype=int))

    # JSON summary (no per-tick arrays)
    summary = {
        'params': {k: v for k, v in params.items()
                   if not isinstance(v, list) or len(v) < 50},
        'geometry': {
            'mode':              'quadratic',
            'quad_alpha':        params['quad_alpha'],
            'quad_center':       params['quad_center'],
            'lower_detector_map': lower_detector_map,
        },
        'by_delta_phi': {
            str(d): {
                'detector_profile': all_runs[d]['detector_profile'],
                'detector_total':   sum(all_runs[d]['detector_profile']),
                'det_blocked':      {k: v for k, v in all_runs[d]['det_blocked'].items()},
                'upper_sends':      all_runs[d]['upper_sends'],
                'lower_sends':      all_runs[d]['lower_sends'],
            }
            for d in range(m)
        },
    }
    with open(os.path.join(outdir, 'results.json'), 'w') as f:
        json.dump(summary, f, indent=2)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    params = DEFAULT_PARAMS.copy()
    m      = params['m']
    n_col  = params['n_columns']

    # Resolve quad_center
    if params['quad_center'] is None:
        params['quad_center'] = (n_col + 1) / 2.0

    # Build representative delta_phi selection automatically from m,
    # but only when not explicitly set in DEFAULT_PARAMS.
    if params['selected_delta_phi'] is None:
        params['selected_delta_phi'] = sorted(
            set([0, 1, m // 4, m // 2, (3 * m) // 4, m - 1]))
    if params['selected_timeline_delta_phi'] is None:
        params['selected_timeline_delta_phi'] = [0, m // 2]

    # Normalise to lists (user may supply a bare int) and clamp to [0, m-1].
    sel = params['selected_delta_phi']
    if isinstance(sel, int):
        sel = [sel]
    params['selected_delta_phi'] = [d for d in sel if 0 <= d < m]

    sel_t = params['selected_timeline_delta_phi']
    if isinstance(sel_t, int):
        sel_t = [sel_t]
    params['selected_timeline_delta_phi'] = [d for d in sel_t if 0 <= d < m]

    # Print geometry summary
    lower_map = build_lower_detector_map(params)
    print(f"m={m}  tau0={params['tau0']}  n_ticks={params['n_ticks']}"
          f"  n_columns={n_col}"
          f"  quad_alpha={params['quad_alpha']:.4f}"
          f"  quad_center={params['quad_center']:.1f}")
    print(f"source_period={params['source_period']}"
          f"  cap_transport={params['capacity_transport']}"
          f"  cap_detector={params['capacity_detector']}")
    print(f"lower_detector_map: {lower_map}")
    print(f"selected_delta_phi: {params['selected_delta_phi']}")
    print()

    # Sweep delta_phi = 0 .. m-1
    all_runs = {}
    for delta in range(m):
        run = run_single(delta, params)
        all_runs[delta] = run
        total = sum(run['detector_profile'])
        # Compact profile string for selected deltas only
        if delta in params['selected_delta_phi']:
            profile_str = '  '.join(f'D{i+1}:{v}' for i, v in
                                    enumerate(run['detector_profile']))
            tag = '  <-- reference' if delta == 0 else (
                  '  <-- half-period' if delta == m // 2 else '')
            print(f"delta_phi={delta:3d}  total={total:5d}  {profile_str}{tag}")
        else:
            print(f"delta_phi={delta:3d}  total={total:5d}")

    print()

    # Geometry-only plot (independent of dynamics)
    plot_lower_detector_map(params, OUTDIR)

    # Main diagnostic plots
    if params['save_full_heatmap']:
        plot_heatmap(all_runs, params, OUTDIR)

    if params['save_selected_profiles']:
        plot_selected_profiles(all_runs, params, OUTDIR)

    if params['save_total_vs_phase']:
        plot_total_vs_phase(all_runs, params, OUTDIR)

    if params['save_timeline']:
        for delta in params['selected_timeline_delta_phi']:
            plot_timeline(all_runs[delta], params, delta, OUTDIR)

    # Suppression profile
    plot_suppression_profile(all_runs, params, OUTDIR)

    save_results(all_runs, params, OUTDIR)

    # Summary statistics
    totals = [sum(all_runs[d]['detector_profile']) for d in range(m)]
    min_total = min(totals)
    max_total = max(totals)
    min_delta = totals.index(min_total)
    max_delta = totals.index(max_total)
    print(f"total range: {min_total} (delta_phi={min_delta})"
          f" .. {max_total} (delta_phi={max_delta})"
          f"  ratio={max_total / max(min_total, 1):.2f}x")
    print(f"results saved to {OUTDIR}")


if __name__ == '__main__':
    main()
