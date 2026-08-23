"""
=============================================================================
d3_compression.py  —  D3: ATP-3 Compression Investigation
=============================================================================
Thesis  : Automation of pyrometer data pre-processing
          (ATP-2 Calibration + ATP-3 Compression)
Author  : [Your Name]

WHAT THIS SCRIPT DOES:
  D3 — Investigation of ML/AI architectures for compression (ATP-3).
  Compares all 3 compression methods across 4 aspects:

    1. CR vs RMSE trade-off  — sweep different compression settings
    2. Reconstruction quality — original vs reconstructed signals
    3. Latent space analysis  — VAE latent dimension effect
    4. Summary metrics        — CR, RMSE, compressed size

OUTPUT:
  d3_compression_comparison.png — 4-panel comparison figure
  d3_compression_summary.csv    — full metrics table

HOW TO RUN:
  python d3_compression.py
  python d3_compression.py /path/to/data/
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

from calibrate import calibrate
from compress  import (delta_encoding, vae, deep_autoencoder,
                       compression_ratio, recon_rmse)

np.random.seed(42)


# =============================================================================
# DATA PREPARATION
# =============================================================================

def prepare_calibrated_signal(data_dir: str) -> tuple:
    """
    Load NIST Layer01, simulate pyrometer + TC, apply best
    calibration (linear) to get a calibrated signal for compression.
    """
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
    T_pyr     = (T_true * (1.0/eps1)**0.25 +
                 np.random.normal(0, 12, n) + spk1)

    # Simulate Thermocouple
    T_tc    = np.zeros(n); T_tc[0] = T_true[0]
    for i in range(1, n):
        T_tc[i] = T_tc[i-1] + 0.08*(T_true[i] - T_tc[i-1])
    T_tc += np.random.normal(0, 2, n)

    # Apply linear calibration (ATP-2 best method)
    T_cal, _ = calibrate(T_pyr, T_tc, method='linear', cal_fraction=0.20)

    print(f"  Layer01: {n} frames | "
          f"T_cal=[{T_cal.min():.0f}, {T_cal.max():.0f}]°C")
    return T_cal, T_true, time_s, n


# =============================================================================
# TRADE-OFF SWEEP FUNCTIONS
# =============================================================================

def sweep_delta_bits(T_cal: np.ndarray) -> pd.DataFrame:
    """
    Sweep Delta Encoding quantisation bits (8 to 16).
    Higher bits → lower RMSE but larger compressed size.
    """
    rows = []
    for bits in [8, 10, 12, 14, 16]:
        res = delta_encoding(T_cal, quantise_bits=bits)
        rows.append({
            'bits'              : bits,
            'compression_ratio' : res['compression_ratio'],
            'recon_rmse'        : res['recon_rmse'],
            'compressed_size'   : res['compressed_size'],
        })
    return pd.DataFrame(rows)


def sweep_vae_latent(T_cal: np.ndarray,
                     window_size: int = 64) -> pd.DataFrame:
    """
    Sweep VAE latent dimensions (2, 4, 8, 16).
    Smaller latent → higher CR but higher RMSE.
    """
    rows = []
    for ld in [2, 4, 8, 16]:
        print(f"    VAE latent_dim={ld}...")
        res = vae(T_cal, window_size=window_size,
                  latent_dim=ld, epochs=50, batch_size=32)
        rows.append({
            'latent_dim'        : ld,
            'compression_ratio' : res['compression_ratio'],
            'recon_rmse'        : res['recon_rmse'],
            'compressed_size'   : res['compressed_size'],
        })
    return pd.DataFrame(rows)


def sweep_deep_ae_bottleneck(T_cal: np.ndarray,
                              window_size: int = 64) -> pd.DataFrame:
    """
    Sweep Deep Autoencoder bottleneck size (2, 4, 8, 16).
    Smaller bottleneck → higher CR but higher RMSE.
    """
    rows = []
    for bn in [2, 4, 8, 16]:
        print(f"    DeepAE bottleneck={bn}...")
        res = deep_autoencoder(T_cal, window_size=window_size,
                               bottleneck=bn, epochs=50, batch_size=32)
        rows.append({
            'bottleneck'        : bn,
            'compression_ratio' : res['compression_ratio'],
            'recon_rmse'        : res['recon_rmse'],
            'compressed_size'   : res['compressed_size'],
        })
    return pd.DataFrame(rows)


# =============================================================================
# MAIN D3 COMPRESSION COMPARISON
# =============================================================================

def run_d3_compression(data_dir: str) -> None:
    """Run full D3 compression investigation and generate figures."""

    print("=" * 65)
    print("  D3 — ATP-3 Compression Investigation")
    print("  Delta Encoding vs VAE vs Deep Autoencoder")
    print("=" * 65)

    # ── Prepare calibrated signal ─────────────────────────────────────
    print("\nPreparing calibrated signal (ATP-2 linear → ATP-3 input)...")
    T_cal, T_true, time_s, n = prepare_calibrated_signal(data_dir)

    # ── Run default comparison (fixed settings) ───────────────────────
    print("\nRunning default compression (window=64, latent/bottleneck=4)...")

    print("  Delta Encoding...")
    res_delta = delta_encoding(T_cal, quantise_bits=12)

    print("  VAE (latent_dim=4)...")
    res_vae   = vae(T_cal, window_size=64, latent_dim=4, epochs=50)

    print("  Deep Autoencoder (bottleneck=4)...")
    res_deep  = deep_autoencoder(T_cal, window_size=64,
                                 bottleneck=4, epochs=50)

    default_results = {
        'Delta Encoding'  : res_delta,
        'VAE'             : res_vae,
        'Deep Autoencoder': res_deep,
    }

    # ── Run trade-off sweeps ──────────────────────────────────────────
    print("\nRunning trade-off sweeps...")

    print("  Delta Encoding bits sweep...")
    df_delta_sweep = sweep_delta_bits(T_cal)

    print("  VAE latent dimension sweep...")
    df_vae_sweep   = sweep_vae_latent(T_cal)

    print("  Deep AE bottleneck sweep...")
    df_deep_sweep  = sweep_deep_ae_bottleneck(T_cal)

    # ── Build summary table ───────────────────────────────────────────
    summary_rows = []
    for name, res in default_results.items():
        mtype = 'Classical' if name == 'Delta Encoding' else 'ML/AI'
        summary_rows.append({
            'Method'            : name,
            'Type'              : mtype,
            'Compression_Ratio' : round(res['compression_ratio'], 2),
            'Recon_RMSE_C'      : round(res['recon_rmse'], 2),
            'Original_Size'     : res['original_size'],
            'Compressed_Size'   : res['compressed_size'],
        })
    df_summary = pd.DataFrame(summary_rows)

    print("\n  Default results:")
    print(df_summary.to_string(index=False))

    # ── Generate 4-panel figure ───────────────────────────────────────
    print("\nGenerating D3 compression figure...")

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle(
        "D3 — ATP-3 Compression Investigation\n"
        "Delta Encoding vs VAE vs Deep Autoencoder | "
        "NIST AMBench IN625 Layer01",
        fontsize=13, fontweight='bold'
    )

    col_map = {
        'Delta Encoding'  : 'navy',
        'VAE'             : 'darkorange',
        'Deep Autoencoder': 'green',
    }

    # ── Panel A — Default CR vs RMSE bar chart ────────────────────────
    ax = axes[0, 0]
    names_d = df_summary['Method'].tolist()
    x_pos   = np.arange(len(names_d))
    bw      = 0.35
    ax2     = ax.twinx()

    bars_cr = ax.bar(x_pos - bw/2,
                     df_summary['Compression_Ratio'],
                     bw, color=[col_map[n] for n in names_d],
                     alpha=0.85, label='Compression Ratio (CR)')
    bars_rm = ax2.bar(x_pos + bw/2,
                      df_summary['Recon_RMSE_C'],
                      bw, color=[col_map[n] for n in names_d],
                      alpha=0.45, label='Recon RMSE (°C)')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(names_d, fontsize=9)
    ax.set_ylabel('Compression Ratio (CR)', color='black')
    ax2.set_ylabel('Reconstruction RMSE (°C)', color='grey')
    ax.set_title('A  Default Settings: CR and RMSE Comparison')
    ax.grid(True, axis='y', alpha=0.3)

    # Annotate
    for bar, val in zip(bars_cr, df_summary['Compression_Ratio']):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.1,
                f'{val:.1f}×', ha='center', va='bottom', fontsize=8)
    for bar, val in zip(bars_rm, df_summary['Recon_RMSE_C']):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.5,
                 f'{val:.1f}°C', ha='center', va='bottom', fontsize=8)

    # ── Panel B — Trade-off sweep: CR vs RMSE ────────────────────────
    ax = axes[0, 1]

    # Delta encoding sweep
    ax.plot(df_delta_sweep['compression_ratio'],
            df_delta_sweep['recon_rmse'],
            'o-', color='navy', lw=1.5, ms=7,
            label='Delta Encoding (bits sweep)')
    for _, row in df_delta_sweep.iterrows():
        ax.annotate(f"{int(row['bits'])}b",
                    (row['compression_ratio'], row['recon_rmse']),
                    textcoords='offset points',
                    xytext=(4, 4), fontsize=7, color='navy')

    # VAE sweep
    ax.plot(df_vae_sweep['compression_ratio'],
            df_vae_sweep['recon_rmse'],
            's-', color='darkorange', lw=1.5, ms=7,
            label='VAE (latent dim sweep)')
    for _, row in df_vae_sweep.iterrows():
        ax.annotate(f"ld={int(row['latent_dim'])}",
                    (row['compression_ratio'], row['recon_rmse']),
                    textcoords='offset points',
                    xytext=(4, 4), fontsize=7, color='darkorange')

    # Deep AE sweep
    ax.plot(df_deep_sweep['compression_ratio'],
            df_deep_sweep['recon_rmse'],
            '^-', color='green', lw=1.5, ms=7,
            label='Deep AE (bottleneck sweep)')
    for _, row in df_deep_sweep.iterrows():
        ax.annotate(f"bn={int(row['bottleneck'])}",
                    (row['compression_ratio'], row['recon_rmse']),
                    textcoords='offset points',
                    xytext=(4, 4), fontsize=7, color='green')

    ax.set_xlabel('Compression Ratio (CR) — higher is better →')
    ax.set_ylabel('Reconstruction RMSE (°C) — lower is better ↓')
    ax.set_title('B  CR vs RMSE Trade-off Sweep')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel C — Reconstructed signals (zoomed) ─────────────────────
    ax = axes[1, 0]
    # Show a 200-sample window around peak temperature
    peak = int(T_cal.argmax())
    lo   = max(0, peak - 100)
    hi   = min(n, peak + 100)
    t_w  = time_s[lo:hi]

    ax.plot(t_w, T_cal[lo:hi], color='black', lw=1.5,
            label='Calibrated (input)', zorder=5)
    for name, res in default_results.items():
        T_r = res['T_reconstructed']
        ax.plot(t_w, T_r[lo:hi], lw=1.0, ls='--',
                color=col_map[name],
                label=f"{name} "
                      f"(CR={res['compression_ratio']:.1f}×, "
                      f"RMSE={res['recon_rmse']:.1f}°C)")
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('C  Reconstructed Signals — Peak Temperature Region')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel D — Reconstruction error over time ──────────────────────
    ax = axes[1, 1]
    for name, res in default_results.items():
        T_r   = res['T_reconstructed']
        n_    = min(n, len(T_r))
        error = np.abs(T_cal[:n_] - T_r[:n_])
        ax.plot(time_s[:n_], error, lw=0.8,
                color=col_map[name],
                label=f"{name} "
                      f"(MAE={error.mean():.1f}°C, "
                      f"max={error.max():.0f}°C)")
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Absolute Reconstruction Error (°C)')
    ax.set_title('D  Reconstruction Error Over Time')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig('d3_compression_comparison.png', dpi=150,
                bbox_inches='tight')
    print("  Saved: d3_compression_comparison.png")
    plt.close()

    # ── Save CSVs ─────────────────────────────────────────────────────
    df_summary.to_csv('d3_compression_summary.csv', index=False)
    print("  Saved: d3_compression_summary.csv")

    # Combined sweep CSV
    df_delta_sweep['method'] = 'Delta Encoding'
    df_delta_sweep['param']  = df_delta_sweep['bits'].astype(str) + ' bits'
    df_vae_sweep['method']   = 'VAE'
    df_vae_sweep['param']    = 'latent=' + df_vae_sweep['latent_dim'].astype(str)
    df_deep_sweep['method']  = 'Deep Autoencoder'
    df_deep_sweep['param']   = 'bottleneck=' + df_deep_sweep['bottleneck'].astype(str)

    df_sweep = pd.concat([
        df_delta_sweep[['method','param','compression_ratio','recon_rmse']],
        df_vae_sweep[['method','param','compression_ratio','recon_rmse']],
        df_deep_sweep[['method','param','compression_ratio','recon_rmse']],
    ], ignore_index=True)
    df_sweep.to_csv('d3_compression_sweep.csv', index=False)
    print("  Saved: d3_compression_sweep.csv")

    # ── Final summary ─────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  D3 ATP-3 RESULTS TABLE")
    print("=" * 65)
    print(df_summary.to_string(index=False))
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
    run_d3_compression(DATA_DIR)
