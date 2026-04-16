"""
01_load_and_clean.py
=====================
Loads and merges three real public datasets for the Dallas-Fort Worth metroplex:

  1. USDA Food Access Research Atlas (Excel)
  2. Census ACS DP03 — Selected Economic Characteristics (CSV)
  3. Census ACS DP05 — Demographics (CSV)

All data is filtered to Census tracts in Dallas (48113), Collin (48085),
Tarrant (48439), and Denton (48121) counties.

Output: data/dfw_tracts_merged.csv
"""

import pandas as pd
import numpy as np
import os
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')

DFW_COUNTY_FIPS = ['48113', '48085', '48439', '48121']
COUNTY_NAMES = {
    '48113': 'Dallas County',
    '48085': 'Collin County',
    '48439': 'Tarrant County',
    '48121': 'Denton County',
}


def load_food_access():
    """
    Load USDA Food Access Research Atlas Excel file.

    Key columns extracted:
      - CensusTract: 11-digit FIPS code
      - LILATracts_1And10: Low-income & low-access at 1mi/10mi (food desert flag)
      - lapop1share: Share of tract population with low food access (1mi)
      - lalowi1share: Share of low-income pop with low food access
      - PovertyRate: Tract poverty rate
      - MedianFamilyIncome: Median family income
    """
    print("\n[1/3] Loading USDA Food Access Research Atlas...")
    path = os.path.join(DATA_DIR, 'food_access_2019.xlsx')

    # The USDA file contains multiple sheets; main tract-level data is in
    # 'Food Access Research Atlas' sheet (or first sheet as fallback)
    try:
        xl = pd.ExcelFile(path, engine='openpyxl')
        sheet_name = None
        for candidate in ['Food Access Research Atlas', 'Sheet1', xl.sheet_names[0]]:
            if candidate in xl.sheet_names:
                sheet_name = candidate
                break
        df = pd.read_excel(path, sheet_name=sheet_name, engine='openpyxl',
                           dtype={'CensusTract': str})
    except Exception as e:
        print(f"  ERROR reading Excel file: {e}")
        sys.exit(1)

    print(f"  Loaded {len(df):,} tracts nationwide")

    # Standardize CensusTract to 11-digit string
    df['CensusTract'] = df['CensusTract'].astype(str).str.replace('.0', '', regex=False).str.zfill(11)
    df['county_fips'] = df['CensusTract'].str[:5]

    # Filter to DFW
    df_dfw = df[df['county_fips'].isin(DFW_COUNTY_FIPS)].copy()
    print(f"  Filtered to {len(df_dfw):,} DFW tracts")

    # Select relevant columns (with fallbacks for column name variations)
    column_map = {
        'CensusTract': 'geoid',
        'county_fips': 'county_fips',
    }

    optional_cols = {
        'LILATracts_1And10': 'food_desert_1mi',
        'LILATracts_halfAnd10': 'food_desert_halfmi',
        'lapop1share': 'pct_low_food_access',
        'lalowi1share': 'pct_lowinc_low_food_access',
        'PovertyRate': 'usda_poverty_rate',
        'MedianFamilyIncome': 'usda_median_family_income',
        'TractSNAP': 'snap_households',
        'lahunv1share': 'pct_no_vehicle_low_access',
    }

    for old, new in optional_cols.items():
        if old in df_dfw.columns:
            column_map[old] = new

    available = [c for c in column_map.keys() if c in df_dfw.columns]
    df_food = df_dfw[available].rename(columns=column_map)

    print(f"  Extracted {len(df_food.columns)} columns: {list(df_food.columns)}")
    return df_food


def find_geoid_column(df):
    """Find the GEOID column in a Census CSV (handles various formats)."""
    for col in ['GEO_ID', 'GEOID', 'geo_id', 'Geography', 'NAME']:
        if col in df.columns:
            return col
    # Fallback: first column
    return df.columns[0]


def extract_tract_geoid(geo_value):
    """
    Extract 11-digit tract GEOID from Census format.

    Census GEO_ID format: '1400000US48113000100' → '48113000100'
    """
    s = str(geo_value)
    if 'US' in s:
        return s.split('US')[-1]
    # If already 11 digits, return as-is
    digits = ''.join(c for c in s if c.isdigit())
    if len(digits) >= 11:
        return digits[-11:]
    return None


def load_census_csv(filename, var_mapping):
    """
    Generic loader for Census ACS CSV files from data.census.gov.

    These files have two header rows: codes and labels. We use the codes.
    """
    path = os.path.join(DATA_DIR, filename)
    df = pd.read_csv(path, dtype=str, low_memory=False)

    # If second row is the label row, drop it
    if len(df) > 0:
        first_val = str(df.iloc[0, 0])
        if 'Geograph' in first_val or 'Estimate' in first_val or 'Label' in first_val:
            df = df.iloc[1:].reset_index(drop=True)

    print(f"  Loaded {len(df):,} rows")

    # Find and parse GEOID
    geoid_col = find_geoid_column(df)
    df['geoid'] = df[geoid_col].apply(extract_tract_geoid)
    df = df[df['geoid'].notna()].copy()
    df['county_fips'] = df['geoid'].str[:5]

    # Filter to DFW
    df_dfw = df[df['county_fips'].isin(DFW_COUNTY_FIPS)].copy()
    print(f"  Filtered to {len(df_dfw):,} DFW tracts")

    # Extract requested variables
    result_cols = ['geoid']
    rename_map = {}
    for code, name in var_mapping.items():
        if code in df_dfw.columns:
            result_cols.append(code)
            rename_map[code] = name
        else:
            print(f"  WARNING: Column {code} ({name}) not found in {filename}")

    df_result = df_dfw[result_cols].rename(columns=rename_map)

    # Convert all non-geoid columns to numeric
    for col in df_result.columns:
        if col != 'geoid':
            df_result[col] = pd.to_numeric(df_result[col], errors='coerce')

    print(f"  Extracted {len(df_result.columns) - 1} variables")
    return df_result


def load_acs_dp03():
    """Load Census ACS DP03 — Selected Economic Characteristics."""
    print("\n[2/3] Loading Census ACS DP03 (Economic Characteristics)...")

    var_mapping = {
        'DP03_0062E': 'median_household_income',
        'DP03_0119PE': 'poverty_rate',
        'DP03_0009PE': 'unemployment_rate',
        'DP03_0025E': 'mean_commute_minutes',
        'DP03_0096PE': 'pct_with_health_insurance',
        'DP03_0128PE': 'pct_below_poverty_all',
    }

    return load_census_csv('acs_dp03.csv', var_mapping)


def load_acs_dp05():
    """Load Census ACS DP05 — Demographics."""
    print("\n[3/3] Loading Census ACS DP05 (Demographics)...")

    var_mapping = {
        'DP05_0001E': 'total_population',
        'DP05_0037PE': 'pct_white',
        'DP05_0038PE': 'pct_black',
        'DP05_0071PE': 'pct_hispanic',
        'DP05_0018E': 'median_age',
    }

    return load_census_csv('acs_dp05.csv', var_mapping)


def load_rent_burden():
    """
    Try to load rent burden from DP04 if available, else compute from DP03.

    DP04_0142PE = % of renter households paying 30%+ of income on rent
    """
    dp04_path = os.path.join(DATA_DIR, 'acs_dp04.csv')
    if os.path.exists(dp04_path):
        print("\n[Bonus] Loading DP04 (Housing) for rent burden...")
        var_mapping = {
            'DP04_0142PE': 'pct_rent_burdened',
            'DP04_0134E': 'median_gross_rent',
        }
        return load_census_csv('acs_dp04.csv', var_mapping)
    return None


def merge_all(df_food, df_econ, df_demo, df_rent):
    """Merge all datasets on geoid."""
    print("\n[Merging] Combining datasets on Census tract GEOID...")

    df = df_food.merge(df_econ, on='geoid', how='outer')
    print(f"  After food + DP03: {len(df):,} tracts")

    df = df.merge(df_demo, on='geoid', how='outer')
    print(f"  After + DP05: {len(df):,} tracts")

    if df_rent is not None:
        df = df.merge(df_rent, on='geoid', how='outer')
        print(f"  After + DP04: {len(df):,} tracts")

    # Recompute county_fips from geoid (in case any rows lost it during merge)
    df['county_fips'] = df['geoid'].astype(str).str[:5]
    df['county_name'] = df['county_fips'].map(COUNTY_NAMES)

    # Keep only DFW tracts
    df = df[df['county_fips'].isin(DFW_COUNTY_FIPS)].copy()

    # Drop tracts with no income data (likely uninhabited)
    if 'median_household_income' in df.columns:
        before = len(df)
        df = df[df['median_household_income'].notna() & (df['median_household_income'] > 0)]
        print(f"  Dropped {before - len(df)} tracts with no income data")

    print(f"\n  ✓ Final merged dataset: {len(df):,} tracts × {len(df.columns)} columns")
    return df


def main():
    df_food = load_food_access()
    df_econ = load_acs_dp03()
    df_demo = load_acs_dp05()
    df_rent = load_rent_burden()

    df = merge_all(df_food, df_econ, df_demo, df_rent)

    out_path = os.path.join(DATA_DIR, 'dfw_tracts_merged.csv')
    df.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}")

    # Print county summary
    print("\n  Tracts by county:")
    for county, count in df['county_name'].value_counts().items():
        print(f"    {county}: {count}")


if __name__ == '__main__':
    main()
