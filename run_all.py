"""
run_all.py
===========
Master script for the Poverty Premium Index project.

Runs the entire pipeline in sequence:
  1. Verify data files exist
  2. Load and clean all three real datasets
  3. Build the composite Poverty Premium Index
  4. Run exploratory data analysis
  5. Run statistical regression
  6. Generate publication-quality figures
  7. Print key findings for your appeal letter

Usage:
    python run_all.py
"""

import os
import sys
import subprocess
import time

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
SRC_DIR = os.path.join(PROJECT_DIR, 'src')
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'outputs')


def print_header(text):
    """Print a styled section header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def check_dependencies():
    """Verify all required Python packages are installed."""
    print_header("STEP 0: Checking Python Dependencies")
    required = ['pandas', 'numpy', 'matplotlib', 'seaborn', 'openpyxl']
    optional = ['statsmodels']

    missing_required = []
    missing_optional = []

    for pkg in required:
        try:
            __import__(pkg)
            print(f"  ✓ {pkg}")
        except ImportError:
            print(f"  ✗ {pkg} — MISSING")
            missing_required.append(pkg)

    for pkg in optional:
        try:
            __import__(pkg)
            print(f"  ✓ {pkg} (optional)")
        except ImportError:
            print(f"  ⚠ {pkg} — MISSING (optional, will use fallback)")
            missing_optional.append(pkg)

    if missing_required:
        print(f"\n  ERROR: Install missing packages with:")
        print(f"    pip install {' '.join(missing_required + missing_optional)}")
        sys.exit(1)


def check_data_files():
    """Verify all three required data files are present."""
    print_header("STEP 1: Checking Data Files")

    required_files = {
        'food_access_2019.xlsx': {
            'description': 'USDA Food Access Research Atlas',
            'url': 'https://www.ers.usda.gov/data-products/food-access-research-atlas/download-the-data',
        },
        'acs_dp03.csv': {
            'description': 'Census ACS DP03 (Economic Characteristics)',
            'url': 'https://data.census.gov/table/ACSDP5Y2022.DP03',
        },
        'acs_dp05.csv': {
            'description': 'Census ACS DP05 (Demographics)',
            'url': 'https://data.census.gov/table/ACSDP5Y2022.DP05',
        },
    }

    missing = []
    for fname, info in required_files.items():
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  ✓ {fname} ({size_mb:.1f} MB) — {info['description']}")
        else:
            print(f"  ✗ {fname} — MISSING")
            missing.append((fname, info))

    if missing:
        print("\n" + "=" * 70)
        print("  MISSING DATA FILES")
        print("=" * 70)
        print("\n  Please download the following files and place them in:")
        print(f"  {DATA_DIR}/\n")
        for fname, info in missing:
            print(f"  📁 {fname}")
            print(f"     {info['description']}")
            print(f"     Download from: {info['url']}\n")
        print("  Then run this script again.")
        sys.exit(1)

    print("\n  ✓ All data files present. Ready to run pipeline.")


def run_script(script_name, step_num, step_name):
    """Run an individual pipeline script and capture output."""
    print_header(f"STEP {step_num}: {step_name}")
    script_path = os.path.join(SRC_DIR, script_name)

    start = time.time()
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=False,
        cwd=PROJECT_DIR,
    )
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\n  ✗ ERROR: {script_name} failed (exit code {result.returncode})")
        sys.exit(1)

    print(f"\n  ✓ Completed in {elapsed:.1f}s")


def print_final_summary():
    """Print final summary with paths to all outputs."""
    print_header("PIPELINE COMPLETE")

    print("\n  Output files:")
    print(f"    📊 Cleaned dataset:    data/dfw_poverty_premium_final.csv")
    print(f"    📈 Figures:            outputs/figures/")
    print(f"    📝 Regression results: outputs/regression_results.txt")
    print(f"    ⭐ Key findings:       outputs/key_findings.txt")

    findings_path = os.path.join(OUTPUT_DIR, 'key_findings.txt')
    if os.path.exists(findings_path):
        print("\n" + "=" * 70)
        print("  KEY FINDINGS FOR YOUR APPEAL LETTER")
        print("=" * 70 + "\n")
        with open(findings_path, 'r') as f:
            print(f.read())


def main():
    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "  THE POVERTY PREMIUM INDEX — DALLAS-FORT WORTH".ljust(68) + "█")
    print("█" + "  Independent Research Pipeline".ljust(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)

    check_dependencies()
    check_data_files()

    run_script('01_load_and_clean.py', 2, 'Loading and Cleaning Real Data')
    run_script('02_build_index.py', 3, 'Building the Poverty Premium Index')
    run_script('03_eda.py', 4, 'Exploratory Data Analysis')
    run_script('04_regression.py', 5, 'Statistical Regression')
    run_script('05_visualizations.py', 6, 'Generating Visualizations')

    print_final_summary()


if __name__ == '__main__':
    main()
