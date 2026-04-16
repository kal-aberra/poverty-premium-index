"""
05_visualizations.py
=====================
Generates publication-quality figures for the Poverty Premium Index project.

Outputs (in outputs/figures/):
  - ppi_distribution.png        — Histogram of PPI by county
  - ppi_vs_income.png           — Scatter: PPI vs median income
  - ppi_by_county.png           — Bar chart: avg PPI per county
  - ppi_quartile_breakdown.png  — Income/race breakdown by PPI quartile
  - race_vs_ppi.png             — Race composition vs PPI
  - correlation_matrix.png      — Heatmap of all variables
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
FIG_DIR = os.path.join(PROJECT_DIR, 'outputs', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

sns.set_theme(style='whitegrid', font_scale=1.1)

COUNTY_COLORS = {
    'Collin County': '#1565C0',
    'Dallas County': '#E53935',
    'Denton County': '#43A047',
    'Tarrant County': '#FF8F00',
}
QUARTILE_COLORS = ['#43A047', '#FDD835', '#FF8F00', '#C62828']


def load_data():
    return pd.read_csv(os.path.join(DATA_DIR, 'dfw_poverty_premium_final.csv'))


def plot_distribution(df):
    if 'county_name' not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    for county, color in COUNTY_COLORS.items():
        subset = df[df['county_name'] == county]['poverty_premium_index']
        if len(subset) > 0:
            ax.hist(subset, bins=25, alpha=0.5, color=color, label=county, edgecolor='white')
    ax.set_xlabel('Poverty Premium Index (PPI)', fontsize=12)
    ax.set_ylabel('Number of Census Tracts', fontsize=12)
    ax.set_title('Distribution of the Poverty Premium Index Across DFW',
                 fontsize=14, fontweight='bold')
    ax.axvline(0, color='black', linewidth=1, linestyle='--', alpha=0.5)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'ppi_distribution.png'), dpi=150)
    plt.close()
    print("  Saved: ppi_distribution.png")


def plot_ppi_vs_income(df):
    if 'median_household_income' not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(11, 7))

    for county, color in COUNTY_COLORS.items():
        subset = df[df['county_name'] == county]
        ax.scatter(subset['median_household_income'], subset['poverty_premium_index'],
                   alpha=0.4, s=20, c=color, label=county, edgecolors='none')

    mask = df['median_household_income'].notna() & df['poverty_premium_index'].notna()
    if mask.sum() > 10:
        z = np.polyfit(df.loc[mask, 'median_household_income'],
                       df.loc[mask, 'poverty_premium_index'], 1)
        p = np.poly1d(z)
        x_range = np.linspace(df['median_household_income'].min(),
                              df['median_household_income'].max(), 100)
        ax.plot(x_range, p(x_range), 'k--', linewidth=2, alpha=0.6)
        r = df['median_household_income'].corr(df['poverty_premium_index'])
        ax.text(0.97, 0.97, f'r = {r:.3f}\nn = {mask.sum()}',
                transform=ax.transAxes, fontsize=11, va='top', ha='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('Median Household Income', fontsize=12)
    ax.set_ylabel('Poverty Premium Index', fontsize=12)
    ax.set_title('The Poverty Premium: Lower Income → Higher Structural Costs',
                 fontsize=14, fontweight='bold')
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'ppi_vs_income.png'), dpi=150)
    plt.close()
    print("  Saved: ppi_vs_income.png")


def plot_county_comparison(df):
    if 'county_name' not in df.columns:
        return
    means = df.groupby('county_name')['poverty_premium_index'].mean().sort_values()

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [COUNTY_COLORS.get(c, '#666') for c in means.index]
    bars = ax.barh(range(len(means)), means.values, color=colors,
                   edgecolor='white', height=0.5)
    for bar, val in zip(bars, means.values):
        ax.text(val + 0.01 if val >= 0 else val - 0.01,
                bar.get_y() + bar.get_height() / 2,
                f'{val:+.3f}', va='center', fontsize=11, fontweight='bold',
                ha='left' if val >= 0 else 'right')

    ax.set_yticks(range(len(means)))
    ax.set_yticklabels(means.index, fontsize=11)
    ax.set_xlabel('Average Poverty Premium Index', fontsize=12)
    ax.set_title('Average Structural Cost Burden by County',
                 fontsize=14, fontweight='bold')
    ax.axvline(0, color='black', linewidth=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'ppi_by_county.png'), dpi=150)
    plt.close()
    print("  Saved: ppi_by_county.png")


def plot_quartile_breakdown(df):
    if 'ppi_quartile' not in df.columns:
        return
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    quartiles = ['Q1 (Lowest Cost)', 'Q2', 'Q3', 'Q4 (Highest Cost)']

    panels = [
        ('median_household_income', 'Median Income', '${:,.0f}', lambda x: x),
        ('poverty_rate', 'Avg Poverty Rate', '{:.1f}%', lambda x: x),
        ('pct_black', 'Avg % Black', '{:.1f}%', lambda x: x),
    ]

    for ax, (col, title, fmt, transform) in zip(axes, panels):
        if col not in df.columns:
            ax.set_visible(False)
            continue
        means = df.groupby('ppi_quartile', observed=True)[col].mean()
        vals = [means.get(q, 0) for q in quartiles]
        bars = ax.bar(range(4), vals, color=QUARTILE_COLORS, edgecolor='white', width=0.6)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    val + max(vals) * 0.02,
                    fmt.format(val), ha='center', fontsize=9, fontweight='bold')
        ax.set_xticks(range(4))
        ax.set_xticklabels(['Q1\nLow Cost', 'Q2', 'Q3', 'Q4\nHigh Cost'], fontsize=9)
        ax.set_title(title, fontsize=12, fontweight='bold')
        if 'income' in col.lower():
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    plt.suptitle('The Poverty Premium: Who Pays the Hidden Tax?',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'ppi_quartile_breakdown.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: ppi_quartile_breakdown.png")


def plot_race_vs_ppi(df):
    if 'pct_black' not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(10, 7))

    if 'income_quartile' in df.columns:
        q_colors = {
            'Q1 (Lowest Income)': '#C62828',
            'Q2': '#FF8F00',
            'Q3': '#1976D2',
            'Q4 (Highest Income)': '#1565C0',
        }
        for q, color in q_colors.items():
            subset = df[df['income_quartile'] == q]
            ax.scatter(subset['pct_black'], subset['poverty_premium_index'],
                       alpha=0.5, s=20, c=color, label=q, edgecolors='none')

    mask = df['pct_black'].notna() & df['poverty_premium_index'].notna()
    if mask.sum() > 10:
        z = np.polyfit(df.loc[mask, 'pct_black'], df.loc[mask, 'poverty_premium_index'], 1)
        p = np.poly1d(z)
        x_range = np.linspace(0, df['pct_black'].max(), 100)
        ax.plot(x_range, p(x_range), 'k--', linewidth=2, alpha=0.6)
        r = df['pct_black'].corr(df['poverty_premium_index'])
        ax.text(0.97, 0.03, f'r = {r:.3f}', transform=ax.transAxes,
                fontsize=11, va='bottom', ha='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('% Black Population', fontsize=12)
    ax.set_ylabel('Poverty Premium Index', fontsize=12)
    ax.set_title('Racial Composition and the Poverty Premium',
                 fontsize=14, fontweight='bold')
    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    ax.legend(fontsize=9, title='Income Quartile')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'race_vs_ppi.png'), dpi=150)
    plt.close()
    print("  Saved: race_vs_ppi.png")


def plot_correlation_matrix(df):
    cols = []
    label_map = {
        'median_household_income': 'Median Income',
        'poverty_rate': 'Poverty Rate',
        'unemployment_rate': 'Unemployment',
        'mean_commute_minutes': 'Commute Time',
        'pct_white': '% White',
        'pct_black': '% Black',
        'pct_hispanic': '% Hispanic',
        'pct_low_food_access': '% Low Food Access',
        'poverty_premium_index': 'PPI (Composite)',
    }
    for c in label_map.keys():
        if c in df.columns:
            cols.append(c)

    if len(cols) < 3:
        return

    corr = df[cols].corr()
    labels = [label_map[c] for c in cols]

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, vmin=-1, vmax=1, square=True,
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5, ax=ax,
                cbar_kws={'shrink': 0.8})
    ax.set_title('Correlation Matrix: PPI Components & Demographics',
                 fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'correlation_matrix.png'), dpi=150)
    plt.close()
    print("  Saved: correlation_matrix.png")


def main():
    df = load_data()
    print(f"Loaded {len(df)} tracts for visualization\n")

    plot_distribution(df)
    plot_ppi_vs_income(df)
    plot_county_comparison(df)
    plot_quartile_breakdown(df)
    plot_race_vs_ppi(df)
    plot_correlation_matrix(df)

    print(f"\n  ✓ All figures saved to {FIG_DIR}")


if __name__ == '__main__':
    main()
