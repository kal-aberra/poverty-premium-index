"""
02_build_index.py
==================
Constructs the Poverty Premium Index (PPI) from the merged dataset.

Components (each z-scored, then averaged):
  1. Food Access Premium    — % of population with low food access (USDA)
  2. Transportation Burden  — mean commute time (Census)
  3. Unemployment Burden    — local unemployment rate (Census)
  4. Rent Burden           — % paying 30%+ of income on rent (Census, optional)

Higher PPI = higher hidden cost of poverty.

Input:  data/dfw_tracts_merged.csv
Output: data/dfw_poverty_premium_final.csv
"""

import pandas as pd
import numpy as np
import os

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')


def z_score(series):
    """Standardize to mean=0, std=1, ignoring NaN."""
    mean = series.mean()
    std = series.std()
    if std == 0 or pd.isna(std):
        return pd.Series(0, index=series.index)
    return (series - mean) / std


def load_data():
    path = os.path.join(DATA_DIR, 'dfw_tracts_merged.csv')
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} tracts from merged dataset\n")
    return df


def build_food_premium(df):
    """Component 1: Food access premium."""
    print("[1/4] Food Access Premium")

    # Try multiple food access columns in order of preference
    candidates = ['pct_low_food_access', 'pct_lowinc_low_food_access',
                  'food_desert_1mi', 'food_desert_halfmi']
    food_var = None
    for c in candidates:
        if c in df.columns and df[c].notna().sum() > 10:
            food_var = c
            break

    if food_var is None:
        print("  ⚠ No food access variable available — setting to 0")
        df['food_premium_z'] = 0
        return df

    df[food_var] = pd.to_numeric(df[food_var], errors='coerce')
    df[food_var] = df[food_var].fillna(df[food_var].median())
    df['food_premium_z'] = z_score(df[food_var])

    print(f"  Variable used: {food_var}")
    print(f"  Mean: {df[food_var].mean():.3f}, Std: {df[food_var].std():.3f}")

    # Binary food desert flag for later analysis
    if 'food_desert_1mi' in df.columns:
        df['is_food_desert'] = df['food_desert_1mi'].fillna(0).astype(int)
    else:
        df['is_food_desert'] = (df[food_var] > df[food_var].quantile(0.75)).astype(int)

    return df


def build_transport_burden(df):
    """Component 2: Transportation burden (commute time)."""
    print("\n[2/4] Transportation Burden")

    if 'mean_commute_minutes' in df.columns and df['mean_commute_minutes'].notna().sum() > 10:
        df['mean_commute_minutes'] = df['mean_commute_minutes'].fillna(df['mean_commute_minutes'].median())
        df['transport_burden_z'] = z_score(df['mean_commute_minutes'])
        print(f"  Mean commute: {df['mean_commute_minutes'].mean():.1f} min")
    else:
        print("  ⚠ No commute data — setting to 0")
        df['transport_burden_z'] = 0

    return df


def build_unemployment_burden(df):
    """Component 3: Unemployment burden."""
    print("\n[3/4] Unemployment Burden")

    if 'unemployment_rate' in df.columns and df['unemployment_rate'].notna().sum() > 10:
        df['unemployment_rate'] = df['unemployment_rate'].fillna(df['unemployment_rate'].median())
        df['unemployment_z'] = z_score(df['unemployment_rate'])
        print(f"  Mean unemployment: {df['unemployment_rate'].mean():.2f}%")
    else:
        print("  ⚠ No unemployment data — setting to 0")
        df['unemployment_z'] = 0

    return df


def build_rent_burden(df):
    """Component 4: Rent burden (optional, requires DP04)."""
    print("\n[4/4] Rent Burden")

    if 'pct_rent_burdened' in df.columns and df['pct_rent_burdened'].notna().sum() > 10:
        df['pct_rent_burdened'] = df['pct_rent_burdened'].fillna(df['pct_rent_burdened'].median())
        df['rent_burden_z'] = z_score(df['pct_rent_burdened'])
        print(f"  Mean rent burden: {df['pct_rent_burdened'].mean():.1f}%")
    else:
        print("  ⚠ No rent burden data (download DP04 to add) — setting to 0")
        df['rent_burden_z'] = 0

    return df


def compute_composite_index(df):
    """Combine all z-scored components into the composite PPI."""
    print("\n[Computing] Composite Poverty Premium Index...")

    components = ['food_premium_z', 'transport_burden_z', 'unemployment_z', 'rent_burden_z']
    active = [c for c in components if c in df.columns and (df[c] != 0).any()]

    print(f"  Active components: {active}")

    df['poverty_premium_index'] = df[active].mean(axis=1).round(3)

    # Quartiles
    df['ppi_quartile'] = pd.qcut(
        df['poverty_premium_index'].rank(method='first'),
        q=4,
        labels=['Q1 (Lowest Cost)', 'Q2', 'Q3', 'Q4 (Highest Cost)']
    )

    if 'median_household_income' in df.columns:
        df['income_quartile'] = pd.qcut(
            df['median_household_income'].rank(method='first'),
            q=4,
            labels=['Q1 (Lowest Income)', 'Q2', 'Q3', 'Q4 (Highest Income)']
        )

    print(f"\n  PPI range: {df['poverty_premium_index'].min():.2f} to {df['poverty_premium_index'].max():.2f}")
    print(f"  PPI mean:  {df['poverty_premium_index'].mean():.3f}")
    print(f"  PPI std:   {df['poverty_premium_index'].std():.3f}")

    return df


def save_final(df):
    out_path = os.path.join(DATA_DIR, 'dfw_poverty_premium_final.csv')
    df.to_csv(out_path, index=False)
    print(f"\n  ✓ Saved: {out_path}")
    print(f"    {len(df)} tracts × {len(df.columns)} columns")


def main():
    df = load_data()
    df = build_food_premium(df)
    df = build_transport_burden(df)
    df = build_unemployment_burden(df)
    df = build_rent_burden(df)
    df = compute_composite_index(df)
    save_final(df)


if __name__ == '__main__':
    main()
