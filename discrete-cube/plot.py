#!/usr/bin/env python3
"""Publication-quality plots for two-cube toy results.

Only the input/output paths are configurable; the figure itself is unchanged.

  python plot.py --eval_csv  results/n4_2000/control_eval_n4_2000.csv \
                 --sweep_csv results/n4_2000/phi_sweep_n4_2000.csv \
                 --out       results/n4_2000/toy_final_v8
"""

import argparse

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Style settings - larger fonts
plt.rcParams.update({
    'font.size': 16,
    'font.family': 'serif',
    'axes.labelsize': 18,
    'axes.titlesize': 18,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'legend.fontsize': 14,
    'figure.dpi': 150,
})

# Exclude dx_sign, rename actor_like to full
BASELINES = ["value_only", "distances", "actor_like", "signs"]
BASELINE_LABELS = {
    "value_only": "value_only",
    "distances": "distances",
    "actor_like": "full",
    "signs": "signs",
}

def load_data(eval_csv, sweep_csv):
    """Load and merge CSVs."""
    eval_df = pd.read_csv(eval_csv)
    sweep_df = pd.read_csv(sweep_csv)

    merged = eval_df.merge(
        sweep_df[['phi_id', 'I_A_Z_given_SV', 'H_A_SV']],
        on='phi_id',
        how='left'
    )
    return merged

def plot_success_rate(df, ax):
    """Left panel: success rate scatter."""

    baselines = df[df['phi_id'].isin(BASELINES)]
    randoms = df[~df['phi_id'].isin(BASELINES + ["dx_sign"])]

    x = randoms['Delta_V'].values
    y = randoms['Delta_A'].values
    c = randoms['success_rate'].values

    vmin, vmax = 0.2, 1.0

    sc = ax.scatter(x, y, c=c, s=120, alpha=0.7, cmap='viridis',
                    vmin=vmin, vmax=vmax, edgecolors='none', zorder=2)

    for _, row in baselines.iterrows():
        ax.scatter(row['Delta_V'], row['Delta_A'],
                   c=[row['success_rate']], s=300, marker='X',
                   cmap='viridis', vmin=vmin, vmax=vmax,
                   edgecolors='black', linewidths=1.5, zorder=5)

        label = BASELINE_LABELS[row['phi_id']]
        offset_x, offset_y = 0.05, 0.008
        if row['phi_id'] == 'actor_like':
            offset_x, offset_y = 0.05, 0.012
        elif row['phi_id'] == 'signs':
            offset_x, offset_y = 0.05, 0.01

        ax.annotate(label,
                    (row['Delta_V'] + offset_x, row['Delta_A'] + offset_y),
                    fontsize=18, family='monospace', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.9))

    cbar = plt.colorbar(sc, ax=ax, shrink=0.85, aspect=25, pad=0.02)
    cbar.set_label('Success rate', fontsize=18)
    cbar.ax.tick_params(labelsize=15)

    ax.set_xlabel(r'$\Delta_V = I(V;G|S,Z)$', fontsize=18)
    ax.set_ylabel(r'$\Delta_A = I(A;G|S,Z)$', fontsize=18)
    ax.set_xlim(-0.05, 1.95)
    ax.set_ylim(-0.02, 0.42)

    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)

    ax.set_title('(a) Control success rate', fontsize=18)

def plot_value_sufficient_mi(df, ax):
    """Right panel: For value-sufficient (ΔV≈0) representations,
    show I(A;Z|S,V) vs success rate."""

    threshold = 0.2
    vs_df = df[(df['Delta_V'] < threshold) & (df['I_A_Z_given_SV'].notna())].copy()
    vs_df = vs_df[vs_df['phi_id'] != 'dx_sign']

    baselines_vs = vs_df[vs_df['phi_id'].isin(BASELINES)]
    randoms_vs = vs_df[~vs_df['phi_id'].isin(BASELINES)]

    x_rand = randoms_vs['I_A_Z_given_SV'].values
    y_rand = randoms_vs['success_rate'].values
    c = randoms_vs['Delta_A'].values

    vmin, vmax = 0, 0.25

    sc = ax.scatter(x_rand, y_rand, c=c, s=120, alpha=0.7, cmap='plasma_r',
                    vmin=vmin, vmax=vmax, edgecolors='none', zorder=2)

    for _, row in baselines_vs.iterrows():
        ax.scatter(row['I_A_Z_given_SV'], row['success_rate'],
                   c=[row['Delta_A']], s=300, marker='X',
                   cmap='plasma_r', vmin=vmin, vmax=vmax,
                   edgecolors='black', linewidths=1.5, zorder=5)

    # Labels for baselines
    vo = baselines_vs[baselines_vs['phi_id'] == 'value_only'].iloc[0]
    al = baselines_vs[baselines_vs['phi_id'] == 'actor_like'].iloc[0]
    dist = baselines_vs[baselines_vs['phi_id'] == 'distances'].iloc[0]

    ax.annotate('value_only', (vo['I_A_Z_given_SV'] + 0.015, vo['success_rate'] + 0.04),
                fontsize=18, family='monospace', fontweight='bold', ha='left',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.9))
    ax.annotate('full', (al['I_A_Z_given_SV'] - 0.015, al['success_rate'] - 0.06),
                fontsize=18, family='monospace', fontweight='bold', ha='right',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.9))
    ax.annotate('distances', (dist['I_A_Z_given_SV'] - 0.015, dist['success_rate'] - 0.05),
                fontsize=18, family='monospace', fontweight='bold', ha='right',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.9))

    cbar = plt.colorbar(sc, ax=ax, shrink=0.85, aspect=25, pad=0.02)
    cbar.set_label(r'$\Delta_A$', fontsize=18)
    cbar.ax.tick_params(labelsize=15)

    ax.set_xlabel(r'$I(A;Z|S,V)$', fontsize=18)
    ax.set_ylabel('Success rate', fontsize=18)
    ax.set_ylim(0.35, 1.05)
    ax.set_xlim(-0.01, 0.26)

    ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)

    ax.set_title(r'(b) Value-sufficient ($\Delta_V < 0.2$)', fontsize=18)

    corr = vs_df['I_A_Z_given_SV'].corr(vs_df['success_rate'])

    print(f"\nValue-sufficient subset: {len(vs_df)} phis")
    print(f"  corr(I(A;Z|S,V), success) = {corr:.4f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_csv", default="control_eval_n4_2000.csv")
    ap.add_argument("--sweep_csv", default="phi_sweep_n4_2000.csv")
    ap.add_argument("--out", default="toy_final_v8",
                    help="output path prefix; .png and .pdf are appended")
    args = ap.parse_args()

    df = load_data(args.eval_csv, args.sweep_csv)

    df_no_dx = df[df['phi_id'] != 'dx_sign']

    print(f"Total phis: {len(df_no_dx)}")
    print(f"corr(success, Delta_A) = {df_no_dx['success_rate'].corr(df_no_dx['Delta_A']):.4f}")
    print(f"corr(success, Delta_V) = {df_no_dx['success_rate'].corr(df_no_dx['Delta_V']):.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    plot_success_rate(df, ax1)
    plot_value_sufficient_mi(df, ax2)

    plt.tight_layout()

    fig.savefig(f'{args.out}.png', dpi=300, bbox_inches='tight')
    fig.savefig(f'{args.out}.pdf', bbox_inches='tight')
    print(f"\nSaved: {args.out}.png, {args.out}.pdf")

    plt.close()

if __name__ == "__main__":
    main()