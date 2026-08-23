"""
=============================================================================
d4_visualisation.py  —  D4: Visualisation Tool
=============================================================================
Thesis  : Automation of pyrometer data pre-processing
          (ATP-2 Calibration + ATP-3 Compression)
Author  : [Your Name]

WHAT THIS SCRIPT DOES:
  D4 — Simple visualisation tool for raw vs processed temperature
  and basic event markers.

  Shows the full pipeline from raw pyrometer signals through
  ATP-2 calibration and ATP-3 compression with:
    - Raw vs calibrated temperature comparison
    - Event markers (laser on/off, peak temperature, phase transitions)
    - All 3 compression method reconstructions
    - Error analysis panels

OUTPUT:
  d4_dashboard.png — main visualisation dashboard
  d4_events.csv    — detected event markers table

HOW TO RUN:
  python d4_visualisation.py
  python d4_visualisation.py /path/to/data/
=============================================================================
"""

import os
import sys
import numpy as np
import scipy.io as sio
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

from calibrate import calibrate
from compress  import delta_encoding, vae, deep_autoencoder

np.random.seed(42)


# =============================================================================
# DATA PREPARATION
# =============================================================================

def prepare_data(data_dir: str) -> dict:
    """
    Load NIST Layer01, simulate 2-pyrometer + TC,
    apply ATP-2 calibration, then ATP-3 compression.
    Returns all signals needed for visualisation.
    """
    # ── Load NIST Layer01 ─────────────────────────────────────────────
    target   = 'Layer01.mat'
    mat_path = None
    for fname in os.listdir(data_dir):
        if fname.endswith(target):
            mat_path = os.path.join(data_dir, fname)
            break
    if mat_path is None:
        raise FileNotFoundError(f"Cannot find {target} in {data_dir}")

    mat   = sio.loadmat(mat_path)
    L     = mat['Layer'][0, 0]
    sh_A  = float(L['SHvariable_A'].flat[0])
    sh_B  = float(L['SHvariable_B'].flat[0])
    raw3d = L['RadiantTemp'].astype(np.float32)
    frame_max = raw3d.max(axis=(0, 1))
    T_true = np.clip(sh_A * frame_max + sh_B - 273.15, 0.0, 3000.0)
    T_true = T_true[T_true > 10.0]
    n      = len(T_true)
    time_s = np.arange(n) * 0.002

    # ── Simulate Pyr1 + Pyr2 + TC ────────────────────────────────────
    # Pyrometer 1
    eps1      = 0.85
    spk1      = np.zeros(n)
    si1       = np.random.choice(n, int(n*0.015), replace=False)
    spk1[si1] = np.random.uniform(200, 600, len(si1))
    T_pyr1    = (T_true*(1.0/eps1)**0.25 +
                 np.random.normal(0, 12, n) + spk1)

    # Pyrometer 2
    eps2      = 0.72
    spk2      = np.zeros(n)
    si2       = np.random.choice(n, int(n*0.020), replace=False)
    spk2[si2] = np.random.uniform(150, 500, len(si2))
    T_pyr2    = (T_true*(1.0/eps2)**0.25 +
                 np.random.normal(0, 18, n) +
                 np.linspace(0, 35, n) + spk2)

    # Thermocouple
    T_tc    = np.zeros(n); T_tc[0] = T_true[0]
    for i in range(1, n):
        T_tc[i] = T_tc[i-1] + 0.08*(T_true[i] - T_tc[i-1])
    T_tc += np.random.normal(0, 2, n)

    # ── ATP-2 Calibration (linear — best method from D3) ─────────────
    T_pyr1_cal, _ = calibrate(T_pyr1, T_tc, method='linear')
    T_pyr2_cal, _ = calibrate(T_pyr2, T_tc, method='linear')

    # Fuse both calibrated pyrometers
    def _gauss(s, sigma=15):
        w = int(4*sigma+1); x = np.arange(-w, w+1)
        k = np.exp(-0.5*(x/sigma)**2); k /= k.sum()
        return np.convolve(s.astype(np.float64), k, mode='same')

    s1 = np.std(T_pyr1_cal - _gauss(T_pyr1_cal)) + 1e-9
    s2 = np.std(T_pyr2_cal - _gauss(T_pyr2_cal)) + 1e-9
    w1, w2  = 1/s1**2, 1/s2**2
    T_fused = (w1*T_pyr1_cal + w2*T_pyr2_cal) / (w1+w2)

    # ── ATP-3 Compression (all 3 methods) ────────────────────────────
    print("  Running Delta Encoding...")
    res_delta = delta_encoding(T_fused, quantise_bits=12)

    print("  Running VAE...")
    res_vae   = vae(T_fused, window_size=64, latent_dim=4, epochs=50)

    print("  Running Deep Autoencoder...")
    res_deep  = deep_autoencoder(T_fused, window_size=64,
                                 bottleneck=4, epochs=50)

    # ── Detect event markers ──────────────────────────────────────────
    events = detect_events(T_fused, time_s)

    print(f"  Data ready: {n} frames, {len(events)} events detected")

    return {
        'T_true'    : T_true,
        'T_pyr1'    : T_pyr1,
        'T_pyr2'    : T_pyr2,
        'T_tc'      : T_tc,
        'T_pyr1_cal': T_pyr1_cal,
        'T_pyr2_cal': T_pyr2_cal,
        'T_fused'   : T_fused,
        'res_delta' : res_delta,
        'res_vae'   : res_vae,
        'res_deep'  : res_deep,
        'time_s'    : time_s,
        'n'         : n,
        'events'    : events,
    }


# =============================================================================
# EVENT DETECTION
# =============================================================================

def detect_events(T: np.ndarray, time_s: np.ndarray) -> list:
    """
    Detect basic temperature events in the signal.

    Events detected:
      - Laser ON  : first frame where T > 20% of max temperature
      - Peak temp : frame with maximum temperature
      - Laser OFF : last frame where T > 20% of max temperature
      - Rapid rise: frames where temperature increases > 50°C in 1 step
      - Rapid drop: frames where temperature drops > 50°C in 1 step

    Parameters
    ----------
    T      : temperature signal (°C)
    time_s : time axis (s)

    Returns
    -------
    list of dicts — each with 'event', 'time_s', 'temp_C', 'index'
    """
    events = []
    threshold = 0.20 * T.max()

    # Laser ON — first frame above threshold
    on_idx = np.where(T > threshold)[0]
    if len(on_idx) > 0:
        events.append({
            'event'  : 'Laser ON',
            'index'  : int(on_idx[0]),
            'time_s' : float(time_s[on_idx[0]]),
            'temp_C' : float(T[on_idx[0]]),
        })

    # Peak temperature
    peak_idx = int(T.argmax())
    events.append({
        'event'  : 'Peak Temperature',
        'index'  : peak_idx,
        'time_s' : float(time_s[peak_idx]),
        'temp_C' : float(T[peak_idx]),
    })

    # Laser OFF — last frame above threshold
    if len(on_idx) > 0:
        events.append({
            'event'  : 'Laser OFF',
            'index'  : int(on_idx[-1]),
            'time_s' : float(time_s[on_idx[-1]]),
            'temp_C' : float(T[on_idx[-1]]),
        })

    # Rapid rise events (dT > 50°C in one step)
    dT       = np.diff(T)
    rise_idx = np.where(dT > 50)[0]
    if len(rise_idx) > 0:
        # Report only the top 3 largest rises
        top3 = rise_idx[np.argsort(dT[rise_idx])[-3:]]
        for idx in sorted(top3):
            events.append({
                'event'  : f'Rapid Rise (+{dT[idx]:.0f}°C)',
                'index'  : int(idx),
                'time_s' : float(time_s[idx]),
                'temp_C' : float(T[idx]),
            })

    # Rapid drop events (dT < -50°C in one step)
    drop_idx = np.where(dT < -50)[0]
    if len(drop_idx) > 0:
        top3 = drop_idx[np.argsort(dT[drop_idx])[:3]]
        for idx in sorted(top3):
            events.append({
                'event'  : f'Rapid Drop ({dT[idx]:.0f}°C)',
                'index'  : int(idx),
                'time_s' : float(time_s[idx]),
                'temp_C' : float(T[idx]),
            })

    # Sort by time
    events.sort(key=lambda e: e['time_s'])
    return events


# =============================================================================
# MAIN VISUALISATION
# =============================================================================

def run_d4_visualisation(data_dir: str) -> None:
    """Generate D4 visualisation dashboard."""

    print("=" * 65)
    print("  D4 — Visualisation Tool")
    print("  Raw vs Processed Temperature + Event Markers")
    print("=" * 65)

    print("\nPreparing all pipeline data...")
    data = prepare_data(data_dir)

    T_true     = data['T_true']
    T_pyr1     = data['T_pyr1']
    T_pyr2     = data['T_pyr2']
    T_tc       = data['T_tc']
    T_pyr1_cal = data['T_pyr1_cal']
    T_pyr2_cal = data['T_pyr2_cal']
    T_fused    = data['T_fused']
    res_delta  = data['res_delta']
    res_vae    = data['res_vae']
    res_deep   = data['res_deep']
    time_s     = data['time_s']
    n          = data['n']
    events     = data['events']

    print(f"\nGenerating D4 dashboard...")

    # ── Build figure with gridspec ────────────────────────────────────
    fig = plt.figure(figsize=(18, 16))
    fig.suptitle(
        "D4 — Visualisation Tool: Raw vs Processed Temperature\n"
        "ATP-2 Calibration + ATP-3 Compression + Event Markers | "
        "NIST AMBench IN625 Layer01",
        fontsize=13, fontweight='bold', y=0.98
    )

    gs = gridspec.GridSpec(3, 2, figure=fig,
                           hspace=0.50, wspace=0.35,
                           top=0.92, bottom=0.07,
                           left=0.08, right=0.97)

    # ── Panel A — Raw signals + event markers ────────────────────────
    ax = fig.add_subplot(gs[0, :])   # full width top row
    ax.plot(time_s, T_pyr1, color='steelblue', lw=0.6,
            alpha=0.50, label='Pyr1 raw (ε=0.85)')
    ax.plot(time_s, T_pyr2, color='darkorange', lw=0.6,
            alpha=0.50, label='Pyr2 raw (ε=0.72, drift)')
    ax.plot(time_s, T_tc,   color='green', lw=1.2,
            label='Thermocouple (reference)')
    ax.plot(time_s, T_true, color='black', lw=1.0,
            ls='--', alpha=0.45, label='True temperature')

    # Add event markers
    event_colours = {
        'Laser ON'       : 'lime',
        'Peak Temperature': 'red',
        'Laser OFF'      : 'darkred',
    }
    for ev in events:
        ename = ev['event']
        if 'Rapid' in ename:
            ax.axvline(ev['time_s'], color='purple',
                       lw=0.8, ls=':', alpha=0.6)
        else:
            col = event_colours.get(ename, 'grey')
            ax.axvline(ev['time_s'], color=col, lw=1.5,
                       ls='--', alpha=0.8)
            ax.annotate(ename,
                        xy=(ev['time_s'], ev['temp_C']),
                        xytext=(ev['time_s']+0.02,
                                ev['temp_C']*0.85),
                        fontsize=7, color=col,
                        arrowprops=dict(arrowstyle='->', color=col,
                                        lw=0.8))

    ax.set_title('A  Raw Input Signals with Event Markers', fontsize=10)
    ax.set_ylabel('Temperature (°C)', fontsize=9)
    ax.set_xlabel('Time (s)', fontsize=9)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)

    # ── Panel B — Raw vs Calibrated comparison ────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(time_s, T_pyr1,     color='steelblue', lw=0.6,
            alpha=0.45, label='Pyr1 raw')
    ax.plot(time_s, T_pyr2,     color='darkorange', lw=0.6,
            alpha=0.45, label='Pyr2 raw')
    ax.plot(time_s, T_pyr1_cal, color='steelblue', lw=1.2,
            label='Pyr1 calibrated (ATP-2)')
    ax.plot(time_s, T_pyr2_cal, color='darkorange', lw=1.2,
            label='Pyr2 calibrated (ATP-2)')
    ax.plot(time_s, T_tc,       color='green', lw=1.2,
            label='TC reference')
    ax.set_title('B  Raw vs Calibrated (ATP-2 Linear)', fontsize=10)
    ax.set_ylabel('Temperature (°C)', fontsize=9)
    ax.set_xlabel('Time (s)', fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # ── Panel C — Fused calibrated signal ────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    fused_rmse = float(np.sqrt(np.mean((T_fused - T_tc)**2)))
    ax.plot(time_s, T_tc,     color='green',  lw=1.2,
            label='TC reference')
    ax.plot(time_s, T_true,   color='black',  lw=1.0,
            ls='--', alpha=0.4, label='True temperature')
    ax.plot(time_s, T_fused,  color='crimson', lw=1.5,
            label=f'Fused calibrated (RMSE={fused_rmse:.1f}°C vs TC)')
    ax.fill_between(time_s,
                    T_fused - fused_rmse,
                    T_fused + fused_rmse,
                    color='crimson', alpha=0.10,
                    label='±1 RMSE band')

    # Mark peak event on this panel too
    peak_ev = next((e for e in events
                    if e['event'] == 'Peak Temperature'), None)
    if peak_ev:
        ax.axvline(peak_ev['time_s'], color='red',
                   lw=1.5, ls='--', alpha=0.7,
                   label=f"Peak: {peak_ev['temp_C']:.0f}°C "
                         f"@ {peak_ev['time_s']:.3f}s")

    ax.set_title('C  Fused Calibrated Signal + Event Marker', fontsize=10)
    ax.set_ylabel('Temperature (°C)', fontsize=9)
    ax.set_xlabel('Time (s)', fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # ── Panel D — ATP-3 compression: all 3 reconstructions ───────────
    ax = fig.add_subplot(gs[2, 0])
    ax.plot(time_s, T_fused, color='black', lw=1.3,
            alpha=0.8, label='Calibrated (ATP-3 input)', zorder=5)

    comp_info = [
        ('Delta Encoding', res_delta, 'navy'),
        ('VAE',            res_vae,   'darkorange'),
        ('Deep AE',        res_deep,  'green'),
    ]
    for name, res, col in comp_info:
        T_r = res['T_reconstructed']
        n_  = min(n, len(T_r))
        ax.plot(time_s[:n_], T_r[:n_], lw=0.9, ls='--',
                color=col, alpha=0.85,
                label=f"{name} "
                      f"CR={res['compression_ratio']:.1f}×, "
                      f"RMSE={res['recon_rmse']:.1f}°C")

    ax.set_title('D  ATP-3: Original vs Reconstructed Signals', fontsize=10)
    ax.set_ylabel('Temperature (°C)', fontsize=9)
    ax.set_xlabel('Time (s)', fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # ── Panel E — Reconstruction error comparison ─────────────────────
    ax = fig.add_subplot(gs[2, 1])
    for name, res, col in comp_info:
        T_r   = res['T_reconstructed']
        n_    = min(n, len(T_r))
        error = np.abs(T_fused[:n_] - T_r[:n_])
        ax.plot(time_s[:n_], error, lw=0.9, color=col,
                label=f"{name} "
                      f"(MAE={error.mean():.1f}°C, "
                      f"max={error.max():.0f}°C)")

    ax.set_title('E  ATP-3: Absolute Reconstruction Error', fontsize=10)
    ax.set_ylabel('|Error| (°C)', fontsize=9)
    ax.set_xlabel('Time (s)', fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # ── Save figure ───────────────────────────────────────────────────
    fig.savefig('d4_dashboard.png', dpi=150, bbox_inches='tight')
    print("  Saved: d4_dashboard.png")
    plt.close()

    # ── Save events CSV ───────────────────────────────────────────────
    df_events = pd.DataFrame(events)
    df_events.to_csv('d4_events.csv', index=False, float_format='%.4f')
    print("  Saved: d4_events.csv")

    # ── Print event summary ───────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  D4 EVENT MARKERS DETECTED")
    print("=" * 65)
    print(df_events[['event','time_s','temp_C']].to_string(index=False))
    print("=" * 65)
    print(f"\n  ATP-3 Compression Summary:")
    for name, res, _ in comp_info:
        print(f"    {name:<20} "
              f"CR={res['compression_ratio']:.1f}×  "
              f"RMSE={res['recon_rmse']:.2f}°C")
    print("=" * 65)


# =============================================================================
# ENTRY POINT
# =============================================================================

def find_default_data_dir():
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data (1)', 'data'),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'),
        os.path.expanduser('~/Downloads/data (1)/data'),
        os.path.expanduser('~/Downloads/data'),
        os.getcwd(),
        '/mnt/user-data/uploads/'
    ]
    for c in candidates:
        if os.path.exists(c):
            for f in os.listdir(c):
                if 'Layer01' in f and f.endswith('.mat'):
                    return c
    return candidates[0]

if __name__ == '__main__':
    DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else find_default_data_dir()
    run_d4_visualisation(DATA_DIR)
