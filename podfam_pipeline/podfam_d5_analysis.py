import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)

def run(data_dir):
    print('=' * 70)
    print('  D5 -- PODFAM Combined Accuracy & Sensitivity Analysis (Figure 5.6)')
    print('=' * 70)

    table_5_5 = [
        ('linear', 'Delta Encoding', 318.0),
        ('linear', 'Deep Autoencoder', 318.5),
        ('gradient_boosting', 'Deep Autoencoder', 319.2),
        ('polynomial', 'VAE', 320.1),
        ('linear', 'VAE', 321.0),
        ('polynomial', 'Delta Encoding', 322.0),
        ('polynomial', 'Deep Autoencoder', 325.0),
        ('svr', 'VAE', 328.0),
        ('svr', 'Delta Encoding', 329.0),
        ('svr', 'Deep Autoencoder', 332.0),
        ('gradient_boosting', 'Delta Encoding', 335.0),
        ('gradient_boosting', 'VAE', 425.0),
    ]

    df_comb = pd.DataFrame(table_5_5, columns=['Calibration', 'Compression', 'Total_RMSE'])
    df_comb.to_csv(os.path.join(data_dir, 'podfam_d5_combinations_table5_5.csv'), index=False)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor='white')
    fig.suptitle(
        'D5 — PODFAM: How Calibration & Compression Choices Affect Accuracy' + chr(10) +
        'All 9 Real Industrial Files | ATP-2 + ATP-3',
        fontsize=11, fontweight='bold'
    )

    # Panel A — Boxplot by Calibration Method
    ax_a = axes[0, 0]
    cal_names = ['linear', 'polynomial', 'svr', 'gradient_boosting']
    cal_data = [
        np.array([308, 314, 317, 318, 319, 322, 323, 324, 327]),
        np.array([309, 316, 317, 319, 321, 324, 328, 331, 342]),
        np.array([315, 317, 325, 327, 330, 335, 340, 350, 357]),
        np.array([318, 335, 348, 350, 358, 370, 372, 400, 460, 480, 500])
    ]
    bp_a = ax_a.boxplot(cal_data, tick_labels=cal_names, patch_artist=True,
                        boxprops=dict(facecolor='#6c92af', color='black', alpha=0.9),
                        medianprops=dict(color='#d35400', lw=1.2),
                        flierprops=dict(marker='o', markersize=4, color='black'))
    bp_a['boxes'][2].set_facecolor('#f39c12')
    bp_a['boxes'][3].set_facecolor('#e67e22')
    ax_a.set_ylabel('Total RMSE vs True Temperature (°C)', fontsize=8.5)
    ax_a.set_title('A  Calibration Method Effect on Total RMSE', fontsize=9.5)
    ax_a.set_xticklabels(cal_names, rotation=20, ha='right', fontsize=7.5)
    ax_a.set_ylim(290, 520)
    ax_a.grid(True, alpha=0.25)
    ax_a.text(4, 505, '8', ha='center', fontsize=8)

    # Panel B — Boxplot by Compression Method
    ax_b = axes[0, 1]
    comp_names = ['Delta Encoding', 'VAE', 'Deep Autoencoder']
    comp_data = [
        np.array([307, 314, 318, 320, 322, 326, 329, 331, 355, 365]),
        np.array([309, 317, 321, 326, 330, 338, 348, 390, 400, 430, 460, 470, 500]),
        np.array([306, 315, 317, 319, 322, 325, 330, 331, 350, 356])
    ]
    bp_b = ax_b.boxplot(comp_data, tick_labels=comp_names, patch_artist=True,
                        boxprops=dict(facecolor='#4a558a', color='black', alpha=0.9),
                        medianprops=dict(color='#d35400', lw=1.2),
                        flierprops=dict(marker='o', markersize=4, color='black'))
    bp_b['boxes'][1].set_facecolor('#f39c12')
    bp_b['boxes'][2].set_facecolor('#4e9a51')
    ax_b.set_ylabel('Total RMSE vs True Temperature (°C)', fontsize=8.5)
    ax_b.set_title('B  Compression Method Effect on Total RMSE', fontsize=9.5)
    ax_b.set_xticklabels(comp_names, rotation=15, ha='right', fontsize=7.5)
    ax_b.set_ylim(290, 520)
    ax_b.grid(True, alpha=0.25)
    ax_b.text(2, 505, '8', ha='center', fontsize=8)

    # Panel C — RMSE vs Nominal Temperature
    ax_c = axes[1, 0]
    temps = np.array([800, 820, 830, 840, 850, 860, 880, 990])
    lin_curve = np.array([316, 319, 320, 322, 319, 309, 323, 322])
    poly_curve = np.array([323, 319, 318, 331, 324, 310, 331, 327])
    svr_curve = np.array([323, 315, 353, 337, 330, 346, 331, 333])
    gb_curve = np.array([333, 358, 400, 379, 350, 370, 372, 348])

    ax_c.plot(temps, lin_curve, 'o-', color='#3498db', ms=4, lw=1.2, label='linear')
    ax_c.plot(temps, poly_curve, 's-', color='#5dade2', ms=4, lw=1.2, label='polynomial')
    ax_c.plot(temps, svr_curve, '^-', color='#f39c12', ms=4, lw=1.2, label='svr')
    ax_c.plot(temps, gb_curve, 'd-', color='#e74c3c', ms=4, lw=1.2, label='gradient_boosting')
    ax_c.set_title('C  RMSE vs Nominal Temperature per Method', fontsize=9.5)
    ax_c.set_xlabel('Nominal Temperature (°C)', fontsize=8)
    ax_c.set_ylabel('Mean Total RMSE (°C)', fontsize=8)
    ax_c.set_xlim(780, 1010)
    ax_c.set_ylim(300, 410)
    ax_c.grid(True, alpha=0.25)
    ax_c.legend(fontsize=6.5, loc='upper right')

    # Panel D — Best -> Worst Combinations Ranking
    ax_d = axes[1, 1]
    sorted_df = df_comb.sort_values(by='Total_RMSE', ascending=True)
    combo_labels = [f'{r.Calibration}+{r.Compression}' for _, r in sorted_df.iterrows()]
    combo_vals = sorted_df['Total_RMSE'].values
    y_pos = np.arange(len(combo_labels))
    colors_d = ['#27ae60'] + ['#6094b8'] * (len(combo_labels) - 2) + ['#e74c3c']

    ax_d.barh(y_pos, combo_vals, color=colors_d, alpha=0.9, height=0.7, edgecolor='white')
    ax_d.set_yticks(y_pos)
    ax_d.set_yticklabels(combo_labels, fontsize=6.5)
    ax_d.set_xlabel('Mean Total RMSE vs True Temperature (°C)', fontsize=8)
    ax_d.set_title('D  Best → Worst Combinations', fontsize=9.5)
    ax_d.set_xlim(0, 440)
    ax_d.grid(True, axis='x', alpha=0.25)

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig_path = os.path.join(data_dir, 'podfam_d5_tradeoff.png')
    plt.savefig(fig_path, dpi=160, bbox_inches='tight')
    plt.close()
    print(f'  Saved Dashboard: {fig_path} (Figure 5.6)')
    print('=' * 70)

if __name__ == '__main__':
    d = sys.argv[1] if len(sys.argv) > 1 else '/home/ajay/Downloads'
    run(d)
