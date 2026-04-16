"""
03_eda.py
==========
Exploratory analysis of the Poverty Premium Index dataset.

Outputs:
  - Console summary statistics
  - outputs/key_findings.txt — headline numbers for the appeal letter
"""

import pandas as pd
import numpy as np
import os

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'outputs')


def load_data():
    return pd.read_csv(os.path.join(DATA_DIR, 'dfw_poverty_premium_final.csv'))


def county_summary(df):
    print("=" * 70)
    print("SUMMARY BY COUNTY")
    print("=" * 70)

    if 'county_name' not in df.columns:
        return

    summary = df.groupby('county_name').agg(
        Tracts=('geoid', 'count'),
        Med_Income=('median_household_income', 'median'),
        Avg_Poverty=('poverty_rate', 'mean'),
        Avg_PPI=('poverty_premium_index', 'mean'),
    ).round(2)
    print(summary.to_string())


def quartile_summary(df):
    print("\n" + "=" * 70)
    print("INCOME QUARTILE → POVERTY PREMIUM")
    print("=" * 70)

    if 'income_quartile' not in df.columns:
        return

    cols = {'poverty_premium_index': 'mean'}
    if 'pct_black' in df.columns:
        cols['pct_black'] = 'mean'
    if 'pct_hispanic' in df.columns:
        cols['pct_hispanic'] = 'mean'
    if 'is_food_desert' in df.columns:
        cols['is_food_desert'] = 'mean'

    summary = df.groupby('income_quartile', observed=True).agg(cols).round(3)
    print(summary.to_string())


def race_analysis(df):
    print("\n" + "=" * 70)
    print("PPI BY RACIAL COMPOSITION")
    print("=" * 70)

    if 'pct_black' not in df.columns:
        return

    # Bin tracts by % Black
    df['pct_black_bin'] = pd.cut(
        df['pct_black'],
        bins=[0, 10, 25, 50, 100],
        labels=['<10%', '10-25%', '25-50%', '50%+'],
    )

    summary = df.groupby('pct_black_bin', observed=True).agg(
        Tracts=('geoid', 'count'),
        Avg_PPI=('poverty_premium_index', 'mean'),
        Med_Income=('median_household_income', 'median'),
    ).round(2)
    print(summary.to_string())


def extract_key_findings(df):
    """Extract the headline numbers for the appeal letter."""
    findings = []
    findings.append("KEY FINDINGS — POVERTY PREMIUM INDEX")
    findings.append("Dallas-Fort Worth Census Tracts")
    findings.append("=" * 70)
    findings.append("")

    # Total tracts analyzed
    n = len(df)
    findings.append(f"SCOPE")
    findings.append(f"   Analyzed {n} Census tracts across 4 DFW counties")
    findings.append(f"   Data sources: USDA Food Access Atlas, Census ACS DP03 + DP05")
    findings.append("")

    # Income quartile gap
    if 'income_quartile' in df.columns:
        q1 = df[df['income_quartile'] == 'Q1 (Lowest Income)']['poverty_premium_index'].mean()
        q4 = df[df['income_quartile'] == 'Q4 (Highest Income)']['poverty_premium_index'].mean()
        gap = q1 - q4

        findings.append(f"HEADLINE FINDING #1: Income -> Poverty Premium")
        findings.append(f"   Lowest-income tracts have an avg PPI of {q1:+.3f}")
        findings.append(f"   Highest-income tracts have an avg PPI of {q4:+.3f}")
        findings.append(f"   Gap: {gap:.3f} standard deviations of structural cost burden")
        findings.append("")

    # Race gap (controlling for income would be in regression)
    if 'pct_black' in df.columns:
        # Tracts with high Black population vs low
        high_black = df[df['pct_black'] > 50]['poverty_premium_index'].mean()
        low_black = df[df['pct_black'] < 10]['poverty_premium_index'].mean()
        race_gap = high_black - low_black

        n_high_black = (df['pct_black'] > 50).sum()
        n_low_black = (df['pct_black'] < 10).sum()

        findings.append(f"HEADLINE FINDING #2: Race -> Poverty Premium")
        findings.append(f"   Tracts with majority Black population (n={n_high_black}): PPI = {high_black:+.3f}")
        findings.append(f"   Tracts with <10% Black population (n={n_low_black}): PPI = {low_black:+.3f}")
        findings.append(f"   Gap: {race_gap:.3f} standard deviations")
        findings.append("")

    # Food desert finding
    if 'is_food_desert' in df.columns:
        n_fd = df['is_food_desert'].sum()
        pct_fd = n_fd / len(df) * 100
        if 'median_household_income' in df.columns:
            fd_inc = df[df['is_food_desert'] == 1]['median_household_income'].median()
            non_fd_inc = df[df['is_food_desert'] == 0]['median_household_income'].median()
            findings.append(f"HEADLINE FINDING #3: Food Deserts")
            findings.append(f"   {n_fd} tracts ({pct_fd:.1f}%) classified as food deserts")
            findings.append(f"   Median income in food deserts: ${fd_inc:,.0f}")
            findings.append(f"   Median income elsewhere: ${non_fd_inc:,.0f}")
            findings.append(f"   Income gap: ${non_fd_inc - fd_inc:,.0f}")
            findings.append("")

    # Cost difference framing for the appeal letter

    findings.append("=" * 70)
    findings.append("Note: For the regression-controlled finding on race vs. income,")
    findings.append("see outputs/regression_results.txt (Model 3).")

    findings_text = "\n".join(findings)

    out_path = os.path.join(OUTPUT_DIR, 'key_findings.txt')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(findings_text)

    print("\n" + findings_text)
    print(f"\n  Saved: {out_path}")


def main():
    df = load_data()
    print(f"Loaded {len(df)} tracts for EDA\n")

    county_summary(df)
    quartile_summary(df)
    race_analysis(df)
    extract_key_findings(df)


if __name__ == '__main__':
    main()
