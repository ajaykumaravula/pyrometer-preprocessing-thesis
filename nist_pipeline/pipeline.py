"""
=============================================================================
pipeline.py  —  D2: Modular Pre-Processing Pipeline
=============================================================================
Thesis  : Automation of pyrometer data pre-processing
          (ATP-2 Calibration + ATP-3 Compression)
Author  : [Your Name]

WHAT THIS SCRIPT DOES:
  Chains ATP-2 and ATP-3 stages into one modular pipeline:

    STAGE 0 — Load NIST data + build 2-pyrometer + TC simulation
    STAGE 1 — ATP-2 Calibration (8 methods, select best)
    STAGE 2 — ATP-3 Compression (3 methods compared)
    STAGE 3 — Save outputs (CSV + summary files)
    STAGE 4 — Visualise (6-panel dashboard PNG)

  NOTE: ATP-1 denoising is Sravya's work and is NOT part of this
  pipeline. Raw signal feeds directly into ATP-2 calibration.

HOW TO RUN:
  python pipeline.py
  python pipeline.py /path/to/data/

OUTPUTS:
  clean_calibrated_data.csv    — all pipeline stages per-sample
  atp2_calibration_summary.csv — 8 calibration methods compared
  atp3_compression_summary.csv — 3 compression methods compared
  pipeline_dashboard.png       — 6-panel visualisation dashboard
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
from matplotlib.patches import Patch

from calibrate import calibrate_all, calibrate, rmse as cal_rmse
from compress  import compress_all

np.random.seed(42)


# =============================================================================
# STAGE 0 — DATA LOADING + SIMULATION
# =============================================================================

def load_nist(data_dir: str, layer_num: int = 1) -> dict:
    """
    Load one NIST AMBench .mat file and extract the peak temperature
    time-series using Sakuma-Hattori conversion.

    Parameters
    ----------
    data_dir  : directory containing LayerXX.mat files
    layer_num : which layer to load (1-10)

    Returns
    -------
    dict — T_celsius, time_s, n, layer name
    """
    target   = f"Layer{layer_num:02d}.mat"
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
    T_raw     = np.clip(sh_A * frame_max + sh_B - 273.15, 0.0, 3000.0)
    mask      = T_raw > 10.0
    T_celsius = T_raw[mask]
    n         = len(T_celsius)
    time_s    = np.arange(n) * 0.002

    print(f"  Loaded Layer{layer_num:02d}: {n} frames, "
          f"T=[{T_celsius.min():.1f}, {T_celsius.max():.1f}]°C")
    return {
        'T_celsius' : T_celsius,
        'time_s'    : time_s,
        'n'         : n,
        'layer'     : f"Layer{layer_num:02d}",
    }


def build_simulation(T_true_C: np.ndarray) -> dict:
    """
    Build physically realistic 2-pyrometer + thermocouple signals.

    Pyrometer 1 — emissivity ε=0.85, ±12°C noise, 1.5% spikes
    Pyrometer 2 — emissivity ε=0.72, ±18°C noise, 35°C drift, 2% spikes
    Thermocouple — RC-lag (α=0.08) + ±2°C noise (clean reference)

    Parameters
    ----------
    T_true_C : ground-truth temperature from NIST (°C)

    Returns
    -------
    dict — T_pyr1_raw, T_pyr2_raw, T_tc
    """
    n = len(T_true_C)

    # Pyrometer 1
    eps1       = 0.85
    spk1       = np.zeros(n)
    si1        = np.random.choice(n, int(n * 0.015), replace=False)
    spk1[si1]  = np.random.uniform(200, 600, len(si1))
    T_pyr1_raw = (T_true_C * (1.0/eps1)**0.25 +
                  np.random.normal(0, 12, n) + spk1)

    # Pyrometer 2
    eps2       = 0.72
    spk2       = np.zeros(n)
    si2        = np.random.choice(n, int(n * 0.020), replace=False)
    spk2[si2]  = np.random.uniform(150, 500, len(si2))
    T_pyr2_raw = (T_true_C * (1.0/eps2)**0.25 +
                  np.random.normal(0, 18, n) +
                  np.linspace(0, 35, n) + spk2)

    # Thermocouple
    T_tc       = np.zeros(n); T_tc[0] = T_true_C[0]
    for i in range(1, n):
        T_tc[i] = T_tc[i-1] + 0.08 * (T_true_C[i] - T_tc[i-1])
    T_tc += np.random.normal(0, 2, n)

    print(f"  Pyr1 : ε=0.85, noise=±12°C, spikes={len(si1)}")
    print(f"  Pyr2 : ε=0.72, noise=±18°C, drift=35°C, spikes={len(si2)}")
    print(f"  TC   : RC-lag α=0.08, noise=±2°C")

    return {
        'T_pyr1_raw' : T_pyr1_raw,
        'T_pyr2_raw' : T_pyr2_raw,
        'T_tc'       : T_tc,
    }


# =============================================================================
# STAGE 3 — SAVE OUTPUTS
# =============================================================================

def save_outputs(time_s, T_pyr1_raw, T_pyr2_raw, T_tc,
                 T_pyr1_best, T_pyr2_best, T_fused,
                 T_true_C, cal_results, comp_results) -> None:
    """Save all pipeline results to CSV files."""

    # Per-sample CSV
    df_main = pd.DataFrame({
        'time_s'            : time_s,
        'pyr1_raw_C'        : T_pyr1_raw,
        'pyr2_raw_C'        : T_pyr2_raw,
        'tc_ref_C'          : T_tc,
        'pyr1_calibrated_C' : T_pyr1_best,
        'pyr2_calibrated_C' : T_pyr2_best,
        'fused_C'           : T_fused,
        'true_C'            : T_true_C,
    })
    df_main.to_csv('clean_calibrated_data.csv', index=False,
                   float_format='%.4f')
    print(f"  Saved: clean_calibrated_data.csv "
          f"({len(df_main)} rows × {len(df_main.columns)} cols)")

    # ATP-2 calibration summary
    cal_rows = [{
        'Method'   : name,
        'RMSE_C'   : round(r['rmse'], 2),
        'MAE_C'    : round(r['mae'], 2),
        'MaxErr_C' : round(r['max_err'], 2),
        'Time_ms'  : round(r['time_ms'], 1),
    } for name, r in cal_results.items()]
    pd.DataFrame(cal_rows).to_csv('atp2_calibration_summary.csv', index=False)
    print(f"  Saved: atp2_calibration_summary.csv ({len(cal_rows)} methods)")

    # ATP-3 compression summary
    comp_rows = [{
        'Method'            : name,
        'Compression_Ratio' : round(r['compression_ratio'], 2),
        'Recon_RMSE_C'      : round(r['recon_rmse'], 2),
        'Original_Size'     : r['original_size'],
        'Compressed_Size'   : r['compressed_size'],
    } for name, r in comp_results.items()]
    pd.DataFrame(comp_rows).to_csv('atp3_compression_summary.csv', index=False)
    print(f"  Saved: atp3_compression_summary.csv ({len(comp_rows)} methods)")


# =============================================================================
# STAGE 4 — VISUALISATION
# =============================================================================

def visualise(time_s, T_pyr1_raw, T_pyr2_raw, T_tc,
              T_pyr1_best, T_pyr2_best, T_fused,
              T_true_C, cal_results, comp_results,
              best_cal_method: str) -> None:
    """
    Generate 6-panel dashboard PNG.

    Panel A — Raw signals (Pyr1, Pyr2, TC, True)
    Panel B — ATP-2: calibrated signals + fused result
    Panel C — ATP-2: RMSE bar chart (all 8 methods)
    Panel D — ATP-2: calibration error vs TC reference
    Panel E — ATP-3: CR vs RMSE trade-off scatter
    Panel F — ATP-3: original vs reconstructed signals
    """
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle(
        "D2 — Modular Pre-Processing Pipeline\n"
        "ATP-2: Calibration (8 methods)  |  ATP-3: Compression (3 methods)\n"
        "NIST AMBench IN625 Layer01 — Simulated 2-Pyrometer + Thermocouple",
        fontsize=12, fontweight='bold'
    )

    # ── Panel A — Raw signals ─────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(time_s, T_pyr1_raw, alpha=0.45, color='steelblue',
            lw=0.6, label='Pyr1 raw (ε=0.85)')
    ax.plot(time_s, T_pyr2_raw, alpha=0.45, color='darkorange',
            lw=0.6, label='Pyr2 raw (ε=0.72, drift)')
    ax.plot(time_s, T_tc,  color='green',  lw=1.2,
            label='Thermocouple (reference)')
    ax.plot(time_s, T_true_C, color='black', lw=1.0,
            ls='--', alpha=0.5, label='True temperature')
    ax.set_title('A  Stage 0 — Raw Input Signals')
    ax.set_ylabel('Temperature (°C)')
    ax.set_xlabel('Time (s)')
    ax.legend(fontsize=8);  ax.grid(True, alpha=0.3)

    # ── Panel B — ATP-2: Calibrated + fused ──────────────────────────
    ax = axes[0, 1]
    fused_rmse = float(np.sqrt(np.mean((T_fused - T_tc)**2)))
    ax.plot(time_s, T_pyr1_best, color='steelblue', lw=0.9,
            alpha=0.7, label='Pyr1 calibrated')
    ax.plot(time_s, T_pyr2_best, color='darkorange', lw=0.9,
            alpha=0.7, label='Pyr2 calibrated')
    ax.plot(time_s, T_fused, color='crimson', lw=1.5,
            label=f'Fused signal (RMSE={fused_rmse:.1f}°C vs TC)')
    ax.plot(time_s, T_tc, color='green', lw=1.2,
            label='Thermocouple reference')
    ax.fill_between(time_s,
                    T_fused - fused_rmse,
                    T_fused + fused_rmse,
                    color='crimson', alpha=0.1, label='±1 RMSE band')
    ax.set_title(f'B  Stage 1 (ATP-2) — Best Calibration: {best_cal_method}')
    ax.set_ylabel('Temperature (°C)')
    ax.set_xlabel('Time (s)')
    ax.legend(fontsize=8);  ax.grid(True, alpha=0.3)

    # ── Panel C — ATP-2: RMSE bar chart ──────────────────────────────
    ax = axes[1, 0]
    names   = list(cal_results.keys())
    rmses   = [cal_results[n]['rmse'] for n in names]
    colours = ['crimson' if n == best_cal_method else 'steelblue'
               for n in names]
    bars = ax.bar(range(len(names)), rmses, color=colours, alpha=0.82)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=35, ha='right', fontsize=8)
    ax.set_ylabel('RMSE vs TC reference (°C)')
    ax.set_title('C  ATP-2 — All 8 Calibration Methods: RMSE Comparison')
    ax.grid(True, axis='y', alpha=0.3)
    for bar, val in zip(bars, rmses):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 1,
                f'{val:.0f}', ha='center', va='bottom', fontsize=7)
    ax.legend(handles=[
        Patch(color='crimson',   label=f'Best: {best_cal_method}'),
        Patch(color='steelblue', label='Other methods'),
    ], fontsize=8)

    # ── Panel D — ATP-2: calibration error over time ──────────────────
    ax = axes[1, 1]
    error_raw    = T_pyr1_raw  - T_tc
    error_cal    = T_pyr1_best - T_tc
    ax.plot(time_s, error_raw, color='steelblue', lw=0.7,
            alpha=0.5, label=f'Pyr1 raw error (RMSE={np.sqrt(np.mean(error_raw**2)):.0f}°C)')
    ax.plot(time_s, error_cal, color='crimson', lw=1.0,
            label=f'Pyr1 calibrated error (RMSE={np.sqrt(np.mean(error_cal**2)):.0f}°C)')
    ax.axhline(0, color='black', lw=1.0, ls='--', alpha=0.5)
    ax.set_title('D  ATP-2 — Calibration Error vs TC Reference Over Time')
    ax.set_ylabel('Error (°C)')
    ax.set_xlabel('Time (s)')
    ax.legend(fontsize=8);  ax.grid(True, alpha=0.3)

    # ── Panel E — ATP-3: CR vs RMSE scatter ──────────────────────────
    ax = axes[2, 0]
    col_map = {
        'Delta Encoding'  : 'navy',
        'VAE'             : 'darkorange',
        'Deep Autoencoder': 'green',
    }
    for name, res in comp_results.items():
        c = col_map.get(name, 'grey')
        ax.scatter(res['compression_ratio'], res['recon_rmse'],
                   s=200, zorder=5, color=c,
                   label=f"{name}  CR={res['compression_ratio']:.1f}×, "
                         f"RMSE={res['recon_rmse']:.1f}°C")
        ax.annotate(name,
                    (res['compression_ratio'], res['recon_rmse']),
                    textcoords='offset points',
                    xytext=(8, 4), fontsize=8, color=c)
    ax.set_xlabel('Compression Ratio (CR) — higher is better →')
    ax.set_ylabel('Reconstruction RMSE (°C) — lower is better ↓')
    ax.set_title('E  ATP-3 — Compression Ratio vs Reconstruction RMSE')
    ax.legend(fontsize=8);  ax.grid(True, alpha=0.3)

    # ── Panel F — ATP-3: original vs reconstructed ────────────────────
    ax = axes[2, 1]
    ax.plot(time_s, T_fused, color='black', lw=1.2, alpha=0.8,
            label='Calibrated (ATP-2 output → ATP-3 input)')
    line_cols = ['navy', 'darkorange', 'green']
    for (name, res), col in zip(comp_results.items(), line_cols):
        T_r = res['T_reconstructed']
        n_  = min(len(time_s), len(T_r))
        ax.plot(time_s[:n_], T_r[:n_], lw=0.9, alpha=0.75,
                color=col, ls='--',
                label=f"{name}  (CR={res['compression_ratio']:.1f}×, "
                      f"RMSE={res['recon_rmse']:.1f}°C)")
    ax.set_title('F  ATP-3 — Original vs Reconstructed Signals')
    ax.set_ylabel('Temperature (°C)')
    ax.set_xlabel('Time (s)')
    ax.legend(fontsize=7);  ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig('pipeline_dashboard.png', dpi=150, bbox_inches='tight')
    print("  Saved: pipeline_dashboard.png")
    plt.close()


# =============================================================================
# MAIN PIPELINE RUNNER
# =============================================================================

def run_pipeline(data_dir: str) -> None:
    """
    Run the full D2 pipeline:
    Load → ATP-2 Calibrate → ATP-3 Compress → Save → Plot
    """
    print("=" * 70)
    print("  D2 — Modular Pre-Processing Pipeline")
    print("  ATP-2 (8 Calibration Methods) + ATP-3 (3 Compression Methods)")
    print("  Raw signal feeds directly into ATP-2 (no ATP-1 denoising)")
    print("=" * 70)

    # ── STAGE 0: Load + simulate ──────────────────────────────────────
    print("\nSTAGE 0 — Data Loading + Simulation")
    print("-" * 70)
    nist       = load_nist(data_dir, layer_num=1)
    sim        = build_simulation(nist['T_celsius'])
    T_true_C   = nist['T_celsius']
    time_s     = nist['time_s']
    T_pyr1_raw = sim['T_pyr1_raw']
    T_pyr2_raw = sim['T_pyr2_raw']
    T_tc       = sim['T_tc']
    n          = nist['n']

    # ── STAGE 1: ATP-2 Calibration ────────────────────────────────────
    print("\nSTAGE 1 — ATP-2 Calibration (raw signal → 8 methods)")
    print("-" * 70)
    # Run all 8 methods on raw Pyr1
    cal_results  = calibrate_all(T_pyr1_raw, T_tc,
                                 cal_fraction=0.20, verbose=True)
    # Select best (lowest RMSE)
    best_method  = min(cal_results, key=lambda k: cal_results[k]['rmse'])
    T_pyr1_best  = cal_results[best_method]['T_cal']

    # Apply best method to Pyr2
    T_pyr2_best, _ = calibrate(T_pyr2_raw, T_tc,
                                method=best_method, cal_fraction=0.20)

    # Inverse-variance weighted fusion of both calibrated pyrometers
    def _gauss(s, sigma=15):
        w = int(4*sigma+1); x = np.arange(-w, w+1)
        k = np.exp(-0.5*(x/sigma)**2); k /= k.sum()
        return np.convolve(s.astype(np.float64), k, mode='same')

    s1 = np.std(T_pyr1_best - _gauss(T_pyr1_best)) + 1e-9
    s2 = np.std(T_pyr2_best - _gauss(T_pyr2_best)) + 1e-9
    w1, w2  = 1/s1**2, 1/s2**2
    T_fused = (w1 * T_pyr1_best + w2 * T_pyr2_best) / (w1 + w2)
    fused_rmse = float(np.sqrt(np.mean((T_fused - T_tc)**2)))
    print(f"\n  ★ Best method : {best_method}")
    print(f"  Fused RMSE   : {fused_rmse:.2f}°C vs TC reference")

    # ── STAGE 2: ATP-3 Compression ────────────────────────────────────
    print("\nSTAGE 2 — ATP-3 Compression (calibrated signal → 3 methods)")
    print("-" * 70)
    comp_results = compress_all(T_fused, verbose=True)

    # ── STAGE 3: Save ─────────────────────────────────────────────────
    print("\nSTAGE 3 — Saving Outputs")
    print("-" * 70)
    save_outputs(time_s, T_pyr1_raw, T_pyr2_raw, T_tc,
                 T_pyr1_best, T_pyr2_best, T_fused,
                 T_true_C, cal_results, comp_results)

    # ── STAGE 4: Visualise ────────────────────────────────────────────
    print("\nSTAGE 4 — Visualisation Dashboard")
    print("-" * 70)
    visualise(time_s, T_pyr1_raw, T_pyr2_raw, T_tc,
              T_pyr1_best, T_pyr2_best, T_fused,
              T_true_C, cal_results, comp_results, best_method)

    # ── Final summary ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  D2 PIPELINE COMPLETE — SUMMARY")
    print("=" * 70)
    print(f"  Dataset        : NIST AMBench Layer01 ({n} frames)")
    print(f"  ATP-2 Best     : {best_method} "
          f"(RMSE={cal_results[best_method]['rmse']:.2f}°C)")
    print(f"  Fused RMSE     : {fused_rmse:.2f}°C vs TC reference")
    print()
    print(f"  ATP-3 Results  :")
    for name, res in comp_results.items():
        print(f"    {name:<22} "
              f"CR={res['compression_ratio']:.1f}×  "
              f"RMSE={res['recon_rmse']:.2f}°C")
    print()
    print(f"  Files saved    :")
    print(f"    clean_calibrated_data.csv")
    print(f"    atp2_calibration_summary.csv")
    print(f"    atp3_compression_summary.csv")
    print(f"    pipeline_dashboard.png")
    print("=" * 70)


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
    run_pipeline(DATA_DIR)
