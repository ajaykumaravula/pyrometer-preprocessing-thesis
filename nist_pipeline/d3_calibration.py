"""
=============================================================================
d3_calibration.py  —  D3: ATP-2 Calibration Investigation
=============================================================================
Thesis  : Automation of pyrometer data pre-processing
          (ATP-2 Calibration + ATP-3 Compression)
Author  : [Your Name]

WHAT THIS SCRIPT DOES:
  D3 — Investigation of ML/AI architectures for calibration (ATP-2).
  Compares all 8 calibration methods across 4 aspects:

    1. Accuracy      — RMSE, MAE, Max Error vs TC reference
    2. Speed         — computation time (ms)
    3. Trade-off     — accuracy vs complexity
    4. Residual plot — calibration error over time (best classical vs best ML)

OUTPUT:
  d3_calibration_comparison.png  — 4-panel comparison figure
  d3_calibration_summary.csv     — full metrics table

HOW TO RUN:
  python d3_calibration.py
  python d3_calibration.py /path/to/data/
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

from calibrate import calibrate_all, METHODS

np.random.seed(42)


# =============================================================================
# DATA PREPARATION
# =============================================================================

def prepare_data(data_dir: str) -> tuple:
    """
    Load NIST Layer01 and build simulated 2-pyrometer + TC signals.
    Raw signal feeds directly into calibration (no ATP-1 denoising).
    """
    # Load NIST Layer01
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

    # Simulate Pyrometer 1 (primary)
    eps1      = 0.85
    spk1      = np.zeros(n)
    si1       = np.random.choice(n, int(n*0.015), replace=False)
    spk1[si1] = np.random.uniform(200, 600, len(si1))
    T_pyr     = T_true * (1.0/eps1)**0.25 + np.random.normal(0, 12, n) + spk1

    # Simulate Thermocouple reference
    T_tc    = np.zeros(n); T_tc[0] = T_true[0]
    for i in range(1, n):
        T_tc[i] = T_tc[i-1] + 0.08*(T_true[i] - T_tc[i-1])
    T_tc += np.random.normal(0, 2, n)

    print(f"  Layer01: {n} frames | "
          f"T=[{T_true.min():.0f}, {T_true.max():.0f}]°C")
    print(f"  Pyr1: ε=0.85, ±12°C noise | TC: RC-lag α=0.08")

    return T_pyr, T_tc, T_true, time_s, n


# =============================================================================
# MAIN D3 CALIBRATION COMPARISON
# =============================================================================

def run_d3_calibration(data_dir: str) -> None:
    """Run full D3 calibration investigation and generate figures."""

    print("=" * 65)
    print("  D3 — ATP-2 Calibration Investigation")
    print("  Classical vs ML/AI methods comparison")
    print("=" * 65)

    # ── Prepare data ──────────────────────────────────────────────────
    print("\nPreparing data...")
    T_pyr, T_tc, T_true, time_s, n = prepare_data(data_dir)

    # ── Run all 8 methods ─────────────────────────────────────────────
    print("\nRunning all 8 calibration methods...")
    results = calibrate_all(T_pyr, T_tc, cal_fraction=0.20, verbose=True)

    # ── Build summary dataframe ───────────────────────────────────────
    rows = []
    for name, res in results.items():
        mtype = 'ML/AI' if name in [
            'random_forest', 'mlp',
            'gradient_boosting', 'svr'
        ] else 'Classical'
        rows.append({
            'Method'   : name,
            'Type'     : mtype,
            'RMSE_C'   : round(res['rmse'], 2),
            'MAE_C'    : round(res['mae'], 2),
            'MaxErr_C' : round(res['max_err'], 2),
            'Time_ms'  : round(res['time_ms'], 1),
        })
    df = pd.DataFrame(rows)

    # Best classical and best ML
    df_cls = df[df['Type'] == 'Classical']
    df_ml  = df[df['Type'] == 'ML/AI']
    best_cls = df_cls.loc[df_cls['RMSE_C'].idxmin(), 'Method']
    best_ml  = df_ml.loc[df_ml['RMSE_C'].idxmin(),  'Method']
    print(f"\n  Best Classical : {best_cls} "
          f"(RMSE={df_cls['RMSE_C'].min():.2f}°C)")
    print(f"  Best ML/AI     : {best_ml} "
          f"(RMSE={df_ml['RMSE_C'].min():.2f}°C)")

    # ── Generate 4-panel figure ───────────────────────────────────────
    print("\nGenerating D3 calibration figure...")

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle(
        "D3 — ATP-2 Calibration Investigation\n"
        "Classical vs ML/AI Methods | NIST AMBench IN625 Layer01",
        fontsize=13, fontweight='bold'
    )

    names    = df['Method'].tolist()
    types    = df['Type'].tolist()
    colours  = ['steelblue' if t == 'Classical' else 'darkorange'
                for t in types]
    x_pos    = np.arange(len(names))

    # ── Panel A — RMSE comparison ─────────────────────────────────────
    ax = axes[0, 0]
    bars = ax.bar(x_pos, df['RMSE_C'], color=colours, alpha=0.85,
                  edgecolor='white', linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=35, ha='right', fontsize=8)
    ax.set_ylabel('RMSE vs TC Reference (°C)')
    ax.set_title('A  RMSE Comparison — All 8 Methods')
    ax.grid(True, axis='y', alpha=0.3)
    for bar, val in zip(bars, df['RMSE_C']):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 2,
                f'{val:.0f}', ha='center', va='bottom', fontsize=7)
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color='steelblue',  label='Classical'),
        Patch(color='darkorange', label='ML/AI'),
    ], fontsize=9)

    # ── Panel B — MAE + MaxError grouped bars ────────────────────────
    ax = axes[0, 1]
    bw = 0.35
    b1 = ax.bar(x_pos - bw/2, df['MAE_C'],    bw,
                color=colours, alpha=0.85, label='MAE')
    b2 = ax.bar(x_pos + bw/2, df['MaxErr_C'], bw,
                color=colours, alpha=0.45, label='Max Error',
                edgecolor=[c for c in colours], linewidth=1.2)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=35, ha='right', fontsize=8)
    ax.set_ylabel('Error (°C)')
    ax.set_title('B  MAE and Maximum Error Comparison')
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(fontsize=9)

    # ── Panel C — Computation time ────────────────────────────────────
    ax = axes[1, 0]
    plot_times = [max(float(t), 0.05) for t in df['Time_ms']]
    bars_t = ax.bar(x_pos, plot_times, color=colours,
                    alpha=0.85, edgecolor='white')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=35, ha='right', fontsize=8)
    ax.set_ylabel('Computation Time (ms)')
    ax.set_title('C  Computational Complexity (Training + Inference)')
    ax.set_yscale('log')
    ax.set_ylim(bottom=0.02, top=max(plot_times) * 6)
    ax.grid(True, axis='y', alpha=0.3)
    for bar, val in zip(bars_t, df['Time_ms']):
        y_val = max(float(val), 0.05) * 1.25
        ax.text(bar.get_x() + bar.get_width()/2,
                y_val,
                f'{val:.1f}', ha='center', va='bottom', fontsize=7)
    ax.legend(handles=[
        Patch(color='steelblue',  label='Classical'),
        Patch(color='darkorange', label='ML/AI'),
    ], fontsize=9)

    # ── Panel D — Calibration error over time (best classical vs ML) ──
    ax = axes[1, 1]
    T_raw_err   = T_pyr - T_tc
    T_cls_err   = results[best_cls]['T_cal'] - T_tc
    T_ml_err    = results[best_ml]['T_cal']  - T_tc

    ax.plot(time_s, T_raw_err, color='grey', lw=0.6,
            alpha=0.5, label=f'Raw error (RMSE={np.sqrt(np.mean(T_raw_err**2)):.0f}°C)')
    ax.plot(time_s, T_cls_err, color='steelblue', lw=1.0,
            label=f'Best Classical: {best_cls} '
                  f'(RMSE={np.sqrt(np.mean(T_cls_err**2)):.0f}°C)')
    ax.plot(time_s, T_ml_err, color='darkorange', lw=1.0,
            label=f'Best ML/AI: {best_ml} '
                  f'(RMSE={np.sqrt(np.mean(T_ml_err**2)):.0f}°C)')
    ax.axhline(0, color='black', lw=1.0, ls='--', alpha=0.6)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Calibration Error vs TC (°C)')
    ax.set_title('D  Residual Error Over Time — Best Classical vs Best ML/AI')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig('d3_calibration_comparison.png', dpi=150,
                bbox_inches='tight')
    print("  Saved: d3_calibration_comparison.png")
    plt.close()

    # ── Save CSV ──────────────────────────────────────────────────────
    df.to_csv('d3_calibration_summary.csv', index=False)
    print("  Saved: d3_calibration_summary.csv")

    # ── Print final table ─────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  D3 ATP-2 RESULTS TABLE")
    print("=" * 65)
    print(df.to_string(index=False))
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
    run_d3_calibration(DATA_DIR)
