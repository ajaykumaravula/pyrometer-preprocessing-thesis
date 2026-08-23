"""
=============================================================================
d5_analysis.py  —  D5: Calibration & Compression Accuracy Analysis
=============================================================================
Thesis  : Automation of pyrometer data pre-processing
          (ATP-2 Calibration + ATP-3 Compression)
Author  : [Your Name]

WHAT THIS SCRIPT DOES:
  D5 — Brief analysis of how calibration and compression choices
  affect temperature accuracy.

  Analysis covers:
    1. Effect of calibration window size on RMSE (all 8 methods)
    2. Effect of calibration method choice on RMSE + MAE
    3. Effect of compression ratio on reconstruction RMSE
    4. Combined effect: calibration error + compression error

OUTPUT:
  d5_analysis.png     — 4-panel analysis figure
  d5_summary.csv      — full analysis results table

HOW TO RUN:
  python d5_analysis.py
  python d5_analysis.py /path/to/data/
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

from calibrate import calibrate, calibrate_all, METHODS
from compress  import delta_encoding, vae, deep_autoencoder

np.random.seed(42)


# =============================================================================
# DATA PREPARATION
# =============================================================================

def prepare_data(data_dir: str) -> tuple:
    """Load NIST Layer01 and simulate 2-pyrometer + TC signals."""
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

    # Simulate Pyrometer 1
    eps1      = 0.85
    spk1      = np.zeros(n)
    si1       = np.random.choice(n, int(n*0.015), replace=False)
    spk1[si1] = np.random.uniform(200, 600, len(si1))
    T_pyr     = (T_true*(1.0/eps1)**0.25 +
                 np.random.normal(0, 12, n) + spk1)

    # Simulate Thermocouple
    T_tc    = np.zeros(n); T_tc[0] = T_true[0]
    for i in range(1, n):
        T_tc[i] = T_tc[i-1] + 0.08*(T_true[i] - T_tc[i-1])
    T_tc += np.random.normal(0, 2, n)

    print(f"  Layer01: {n} frames | "
          f"T=[{T_true.min():.0f}, {T_true.max():.0f}]°C")
    return T_pyr, T_tc, T_true, time_s, n


# =============================================================================
# ANALYSIS 1 — EFFECT OF CALIBRATION WINDOW SIZE
# =============================================================================

def analyse_window_size(T_pyr: np.ndarray,
                        T_tc: np.ndarray) -> pd.DataFrame:
    """
    Analyse how the calibration window size (fraction of data used
    for fitting) affects RMSE for all 8 methods.

    Window fractions tested: 5%, 10%, 15%, 20%, 30%, 40%, 50%
    """
    fractions = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    rows = []

    # Only test fast methods for the sweep (skip MLP/RF for speed)
    sweep_methods = ['mean_offset', 'linear', 'polynomial',
                     'piecewise_linear', 'svr', 'gradient_boosting']

    for frac in fractions:
        for method in sweep_methods:
            try:
                T_cal, _ = calibrate(T_pyr, T_tc,
                                     method=method,
                                     cal_fraction=frac)
                r = float(np.sqrt(np.mean((T_cal - T_tc)**2)))
                m = float(np.mean(np.abs(T_cal - T_tc)))
            except Exception:
                r, m = np.nan, np.nan
            rows.append({
                'cal_fraction' : frac,
                'cal_pct'      : int(frac * 100),
                'method'       : method,
                'rmse'         : round(r, 2),
                'mae'          : round(m, 2),
            })

    return pd.DataFrame(rows)


# =============================================================================
# ANALYSIS 2 — CALIBRATION METHOD COMPARISON (fixed window=20%)
# =============================================================================

def analyse_calibration_methods(T_pyr: np.ndarray,
                                 T_tc: np.ndarray) -> pd.DataFrame:
    """
    Compare all 8 calibration methods at fixed 20% calibration window.
    Reports RMSE, MAE, MaxError and computation time.
    """
    results = calibrate_all(T_pyr, T_tc, cal_fraction=0.20, verbose=False)
    rows = []
    for name, res in results.items():
        mtype = ('ML/AI' if name in
                 ['random_forest','mlp','gradient_boosting','svr']
                 else 'Classical')
        rows.append({
            'method'   : name,
            'type'     : mtype,
            'rmse'     : round(res['rmse'], 2),
            'mae'      : round(res['mae'], 2),
            'max_err'  : round(res['max_err'], 2),
            'time_ms'  : round(res['time_ms'], 1),
        })
    return pd.DataFrame(rows)


# =============================================================================
# ANALYSIS 3 — EFFECT OF COMPRESSION SETTINGS ON RMSE
# =============================================================================

def analyse_compression_accuracy(T_fused: np.ndarray) -> pd.DataFrame:
    """
    Analyse how compression settings affect reconstruction RMSE.

    Delta Encoding : sweep quantisation bits (8, 10, 12, 14, 16)
    VAE            : sweep latent dimensions (2, 4, 8, 16)
    Deep AE        : sweep bottleneck size (2, 4, 8, 16)
    """
    rows = []

    # Delta Encoding — bits sweep
    print("  Delta Encoding bits sweep...")
    for bits in [8, 10, 12, 14, 16]:
        res = delta_encoding(T_fused, quantise_bits=bits)
        rows.append({
            'method'  : 'Delta Encoding',
            'param'   : f'{bits} bits',
            'param_val': bits,
            'cr'      : round(res['compression_ratio'], 2),
            'rmse'    : round(res['recon_rmse'], 2),
        })

    # VAE — latent dim sweep
    print("  VAE latent dim sweep...")
    for ld in [2, 4, 8, 16]:
        res = vae(T_fused, window_size=64, latent_dim=ld, epochs=50)
        rows.append({
            'method'   : 'VAE',
            'param'    : f'latent={ld}',
            'param_val': ld,
            'cr'       : round(res['compression_ratio'], 2),
            'rmse'     : round(res['recon_rmse'], 2),
        })

    # Deep AE — bottleneck sweep
    print("  Deep AE bottleneck sweep...")
    for bn in [2, 4, 8, 16]:
        res = deep_autoencoder(T_fused, window_size=64,
                               bottleneck=bn, epochs=50)
        rows.append({
            'method'   : 'Deep Autoencoder',
            'param'    : f'bn={bn}',
            'param_val': bn,
            'cr'       : round(res['compression_ratio'], 2),
            'rmse'     : round(res['recon_rmse'], 2),
        })

    return pd.DataFrame(rows)


# =============================================================================
# ANALYSIS 4 — COMBINED CALIBRATION + COMPRESSION ERROR
# =============================================================================

def analyse_combined_error(T_pyr: np.ndarray,
                            T_tc: np.ndarray,
                            T_true: np.ndarray) -> pd.DataFrame:
    """
    Analyse the combined effect of calibration + compression on
    final temperature accuracy vs ground truth.

    For each combination of (calibration method, compression method),
    compute the total RMSE vs true temperature.
    """
    rows = []

    # Use fast calibration methods only
    cal_methods  = ['linear', 'polynomial', 'svr', 'gradient_boosting']
    comp_methods = [
        ('Delta Encoding', lambda T: delta_encoding(T, quantise_bits=12)),
        ('VAE',            lambda T: vae(T, window_size=64, latent_dim=4,
                                         epochs=30)),
        ('Deep AE',        lambda T: deep_autoencoder(T, window_size=64,
                                                       bottleneck=4,
                                                       epochs=30)),
    ]

    for cal_name in cal_methods:
        # Step 1: calibrate
        T_cal, _ = calibrate(T_pyr, T_tc,
                             method=cal_name, cal_fraction=0.20)
        cal_rmse = float(np.sqrt(np.mean((T_cal - T_true)**2)))

        for comp_name, comp_fn in comp_methods:
            # Step 2: compress + reconstruct
            res      = comp_fn(T_cal)
            T_recon  = res['T_reconstructed']
            n_       = min(len(T_true), len(T_recon))
            total_rmse = float(np.sqrt(np.mean(
                (T_true[:n_] - T_recon[:n_])**2
            )))
            rows.append({
                'calibration'       : cal_name,
                'compression'       : comp_name,
                'cal_rmse_vs_true'  : round(cal_rmse, 2),
                'comp_cr'           : round(res['compression_ratio'], 2),
                'comp_recon_rmse'   : round(res['recon_rmse'], 2),
                'total_rmse_vs_true': round(total_rmse, 2),
            })
            print(f"    {cal_name} + {comp_name}: "
                  f"total RMSE={total_rmse:.1f}°C")

    return pd.DataFrame(rows)


# =============================================================================
# MAIN D5 ANALYSIS
# =============================================================================

def run_d5_analysis(data_dir: str) -> None:
    """Run full D5 analysis and generate figures."""

    print("=" * 65)
    print("  D5 — Calibration & Compression Accuracy Analysis")
    print("=" * 65)

    # ── Prepare data ──────────────────────────────────────────────────
    print("\nPreparing data...")
    T_pyr, T_tc, T_true, time_s, n = prepare_data(data_dir)

    # Get best calibrated signal for compression analysis
    T_cal_best, _ = calibrate(T_pyr, T_tc,
                              method='linear', cal_fraction=0.20)

    # ── Run all 4 analyses ────────────────────────────────────────────
    print("\nAnalysis 1 — Calibration window size effect...")
    df_window = analyse_window_size(T_pyr, T_tc)

    print("\nAnalysis 2 — Calibration method comparison...")
    df_methods = analyse_calibration_methods(T_pyr, T_tc)

    print("\nAnalysis 3 — Compression accuracy sweep...")
    df_compress = analyse_compression_accuracy(T_cal_best)

    print("\nAnalysis 4 — Combined calibration + compression error...")
    df_combined = analyse_combined_error(T_pyr, T_tc, T_true)

    # ── Generate 4-panel figure ───────────────────────────────────────
    print("\nGenerating D5 analysis figure...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.subplots_adjust(top=0.90, bottom=0.12,
                        hspace=0.50, wspace=0.35)
    fig.suptitle(
        "D5 — Analysis: How Calibration & Compression Choices "
        "Affect Temperature Accuracy\nNIST AMBench IN625 Layer01",
        fontsize=13, fontweight='bold', y=0.97
    )

    # ── Panel A — Window size effect ──────────────────────────────────
    ax = axes[0, 0]
    method_colours = {
        'mean_offset'     : 'grey',
        'linear'          : 'steelblue',
        'polynomial'      : 'royalblue',
        'piecewise_linear': 'cornflowerblue',
        'svr'             : 'darkorange',
        'gradient_boosting': 'orangered',
    }
    fracs = sorted(df_window['cal_fraction'].unique())
    for method in df_window['method'].unique():
        sub  = df_window[df_window['method'] == method]
        sub  = sub.sort_values('cal_fraction')
        col  = method_colours.get(method, 'black')
        ls   = '--' if method in ['svr','gradient_boosting'] else '-'
        ax.plot(sub['cal_pct'], sub['rmse'],
                marker='o', ms=5, lw=1.4, ls=ls,
                color=col, label=method)
    ax.set_xlabel('Calibration Window Size (%)', fontsize=9)
    ax.set_ylabel('RMSE vs TC Reference (°C)', fontsize=9)
    ax.set_title('A  Effect of Calibration Window Size on RMSE', fontsize=10)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # ── Panel B — Calibration method RMSE + time trade-off ───────────
    ax  = axes[0, 1]
    ax2 = ax.twinx()
    names_m  = df_methods['method'].tolist()
    rmses_m  = df_methods['rmse'].tolist()
    times_m  = df_methods['time_ms'].tolist()
    types_m  = df_methods['type'].tolist()
    cols_m   = ['steelblue' if t == 'Classical' else 'darkorange'
                for t in types_m]
    x_m      = np.arange(len(names_m))
    bw       = 0.35

    b1 = ax.bar(x_m - bw/2, rmses_m, bw,
                color=cols_m, alpha=0.85, label='RMSE (°C)')
    b2 = ax2.bar(x_m + bw/2, times_m, bw,
                 color=cols_m, alpha=0.40, label='Time (ms)')
    ax.set_xticks(x_m)
    ax.set_xticklabels(names_m, rotation=35, ha='right', fontsize=8)
    ax.set_ylabel('RMSE vs TC Reference (°C)', fontsize=9)
    ax2.set_ylabel('Computation Time (ms)', fontsize=9)
    ax.set_title('B  Calibration Method: Accuracy vs Speed Trade-off',
                 fontsize=10)
    ax2.set_yscale('log')
    ax.grid(True, axis='y', alpha=0.3)
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color='steelblue',  label='Classical'),
        Patch(color='darkorange', label='ML/AI'),
    ], fontsize=8, loc='upper left')

    # ── Panel C — Compression CR vs RMSE trade-off ────────────────────
    ax = axes[1, 0]
    comp_colours = {
        'Delta Encoding'  : 'navy',
        'VAE'             : 'darkorange',
        'Deep Autoencoder': 'green',
    }
    comp_markers = {
        'Delta Encoding'  : 'o',
        'VAE'             : 's',
        'Deep Autoencoder': '^',
    }
    for method in df_compress['method'].unique():
        sub = df_compress[df_compress['method'] == method]
        sub = sub.sort_values('cr')
        col = comp_colours.get(method, 'grey')
        mk  = comp_markers.get(method, 'o')
        ax.plot(sub['cr'], sub['rmse'],
                marker=mk, ms=7, lw=1.5,
                color=col, label=method)
        # Annotate each point with its parameter
        for _, row in sub.iterrows():
            ax.annotate(row['param'],
                        (row['cr'], row['rmse']),
                        textcoords='offset points',
                        xytext=(4, 4), fontsize=6,
                        color=col)
    ax.set_xlabel('Compression Ratio (CR) — higher is better →',
                  fontsize=9)
    ax.set_ylabel('Reconstruction RMSE (°C) — lower is better ↓',
                  fontsize=9)
    ax.set_title('C  Compression: CR vs RMSE Trade-off',
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel D — Combined calibration + compression error ────────────
    ax = axes[1, 1]
    comp_methods_d = df_combined['compression'].unique()
    cal_methods_d  = df_combined['calibration'].unique()
    x_d   = np.arange(len(cal_methods_d))
    bw_d  = 0.25
    offsets = np.linspace(-(len(comp_methods_d)-1)*bw_d/2,
                           (len(comp_methods_d)-1)*bw_d/2,
                           len(comp_methods_d))
    comp_cols_d = ['navy', 'darkorange', 'green']

    for i, (comp, col) in enumerate(zip(comp_methods_d, comp_cols_d)):
        sub    = df_combined[df_combined['compression'] == comp]
        sub    = sub.set_index('calibration').reindex(cal_methods_d)
        values = sub['total_rmse_vs_true'].values
        ax.bar(x_d + offsets[i], values, bw_d,
               color=col, alpha=0.82,
               label=comp)

    ax.set_xticks(x_d)
    ax.set_xticklabels(cal_methods_d, rotation=25,
                       ha='right', fontsize=8)
    ax.set_ylabel('Total RMSE vs True Temperature (°C)', fontsize=9)
    ax.set_title('D  Combined Effect: Calibration + Compression vs True Temp',
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, axis='y', alpha=0.3)

    fig.savefig('d5_analysis.png', dpi=150, bbox_inches='tight')
    print("  Saved: d5_analysis.png")
    plt.close()

    # ── Save CSV ──────────────────────────────────────────────────────
    df_combined.to_csv('d5_summary.csv', index=False)
    print("  Saved: d5_summary.csv")

    # ── Print summary ─────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  D5 COMBINED CALIBRATION + COMPRESSION RESULTS")
    print("=" * 65)
    print(df_combined.to_string(index=False))
    print("=" * 65)

    # Key findings
    best_row = df_combined.loc[
        df_combined['total_rmse_vs_true'].idxmin()
    ]
    worst_row = df_combined.loc[
        df_combined['total_rmse_vs_true'].idxmax()
    ]
    print(f"\n  Best combination : "
          f"{best_row['calibration']} + {best_row['compression']} "
          f"→ RMSE={best_row['total_rmse_vs_true']:.1f}°C")
    print(f"  Worst combination: "
          f"{worst_row['calibration']} + {worst_row['compression']} "
          f"→ RMSE={worst_row['total_rmse_vs_true']:.1f}°C")
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
    run_d5_analysis(DATA_DIR)
