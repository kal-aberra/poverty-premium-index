"""
04_regression.py
=================
Statistical regression analysis of the Poverty Premium Index.

Models:
  1. PPI ~ Income (bivariate baseline)
  2. PPI ~ Income + Race + County (multivariate)
  3. PPI ~ Income + Race (KEY TEST: does race matter after controlling for income?)

Outputs:
  - outputs/regression_results.txt — full statistical output
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'outputs')

try:
    import statsmodels.api as sm
    HAS_SM = True
except ImportError:
    HAS_SM = False
    print("WARNING: statsmodels not installed. Using manual OLS fallback.")


def load_data():
    return pd.read_csv(os.path.join(DATA_DIR, 'dfw_poverty_premium_final.csv'))


def manual_ols(X, y):
    """Fallback OLS without statsmodels."""
    X_mat = np.column_stack([np.ones(len(X)), X])
    beta = np.linalg.lstsq(X_mat, y, rcond=None)[0]
    y_pred = X_mat @ beta
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot
    n, k = X_mat.shape
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)
    return {'r2': r2, 'adj_r2': adj_r2, 'beta': beta, 'n': n}


def model_1_bivariate(df, output_lines):
    """PPI ~ Income"""
    output_lines.append("\n" + "=" * 70)
    output_lines.append("MODEL 1: PPI ~ Median Household Income (Bivariate)")
    output_lines.append("=" * 70)

    if 'median_household_income' not in df.columns:
        output_lines.append("Skipped — no income data")
        return None

    mask = df['median_household_income'].notna() & df['poverty_premium_index'].notna()
    X = df.loc[mask, ['median_household_income']]
    y = df.loc[mask, 'poverty_premium_index']

    if HAS_SM:
        Xc = sm.add_constant(X)
        model = sm.OLS(y, Xc).fit()
        output_lines.append(str(model.summary()))
        return model
    else:
        result = manual_ols(X.values, y.values)
        output_lines.append(f"R² = {result['r2']:.4f}, Adj R² = {result['adj_r2']:.4f}")
        output_lines.append(f"Coefficient (income): {result['beta'][1]:.6f}")
        return result


def model_2_full(df, output_lines):
    """PPI ~ Income + Race + County"""
    output_lines.append("\n" + "=" * 70)
    output_lines.append("MODEL 2: PPI ~ Income + Race + County (Full Model)")
    output_lines.append("=" * 70)

    features = []
    for col in ['median_household_income', 'pct_black', 'pct_hispanic', 'poverty_rate']:
        if col in df.columns:
            features.append(col)

    if len(features) < 2:
        output_lines.append("Skipped — insufficient variables")
        return None

    X = df[features].copy()

    # Add county dummies
    if 'county_name' in df.columns:
        dummies = pd.get_dummies(df['county_name'], prefix='county', drop_first=True, dtype=float)
        X = pd.concat([X, dummies], axis=1)

    y = df['poverty_premium_index']
    mask = X.notna().all(axis=1) & y.notna()
    X = X.loc[mask]
    y = y.loc[mask]

    if HAS_SM:
        Xc = sm.add_constant(X)
        model = sm.OLS(y, Xc).fit()
        output_lines.append(str(model.summary()))
        return model
    else:
        result = manual_ols(X.values, y.values)
        output_lines.append(f"R² = {result['r2']:.4f}, Adj R² = {result['adj_r2']:.4f}")
        return result


def model_3_race_test(df, output_lines):
    """KEY TEST: Does race predict PPI after controlling for income?"""
    output_lines.append("\n" + "=" * 70)
    output_lines.append("MODEL 3: KEY TEST — Race Effect After Controlling for Income")
    output_lines.append("=" * 70)
    output_lines.append("")
    output_lines.append("Question: Does racial composition predict the poverty premium")
    output_lines.append("even after accounting for household income?")
    output_lines.append("")

    if 'pct_black' not in df.columns or 'median_household_income' not in df.columns:
        output_lines.append("Skipped — missing required variables")
        return None

    X = df[['median_household_income', 'pct_black']].copy()
    y = df['poverty_premium_index']
    mask = X.notna().all(axis=1) & y.notna()
    X = X.loc[mask]
    y = y.loc[mask]

    if HAS_SM:
        Xc = sm.add_constant(X)
        model = sm.OLS(y, Xc).fit()
        output_lines.append(str(model.summary()))

        # Interpret the key finding
        pct_black_pval = model.pvalues.get('pct_black', None)
        pct_black_coef = model.params.get('pct_black', None)

        output_lines.append("")
        output_lines.append("─" * 70)
        output_lines.append("INTERPRETATION:")
        if pct_black_pval is not None and pct_black_pval < 0.05:
            output_lines.append(f"  YES — % Black IS significant after controlling for income")
            output_lines.append(f"    Coefficient: {pct_black_coef:.4f}")
            output_lines.append(f"    p-value: {pct_black_pval:.4f}")
            output_lines.append(f"    A 10pp increase in % Black is associated with a")
            output_lines.append(f"    {pct_black_coef * 10:+.3f} change in PPI at the same income level.")
            output_lines.append(f"  This suggests structural barriers BEYOND income alone.")
        elif pct_black_pval is not None:
            output_lines.append(f"  NO — % Black NOT significant after controlling for income")
            output_lines.append(f"    p-value: {pct_black_pval:.4f}")
            output_lines.append(f"  Income alone may explain most of the poverty premium.")
        return model
    else:
        result = manual_ols(X.values, y.values)
        output_lines.append(f"R² = {result['r2']:.4f}")
        return result


def main():
    df = load_data()
    print(f"Loaded {len(df)} tracts for regression\n")

    output_lines = []
    output_lines.append("POVERTY PREMIUM INDEX — REGRESSION RESULTS")
    output_lines.append(f"Dallas-Fort Worth Census Tracts (n = {len(df)})")
    output_lines.append("=" * 70)

    model_1_bivariate(df, output_lines)
    model_2_full(df, output_lines)
    model_3_race_test(df, output_lines)

    # Save full output
    out_path = os.path.join(OUTPUT_DIR, 'regression_results.txt')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(output_lines))

    # Print to console too
    print("\n".join(output_lines))
    print(f"\n  Saved: {out_path}")


if __name__ == '__main__':
    main()
