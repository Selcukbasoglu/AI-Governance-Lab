import sys
import os
import pandas as pd
from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import data_integrity

# --- Configuration ---
BIAS_THRESHOLD = 0.35  # Female representation must be > 35%
REPORT_PATH = os.path.join("reports", "integrity_report.html")

# --- 1. DATA LOADING ---
print("⏳ Loading UCI Adult Dataset...")
try:
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    columns = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", 
               "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", 
               "hours-per-week", "native-country", "income"]
    df = pd.read_csv(url, names=columns, na_values=" ?", skipinitialspace=True)
except Exception as e:
    print(f"❌ ERROR: Failed to load data. Check internet or URL. {e}")
    sys.exit(1)

# --- 2. BIAS CHECK (EU AI ACT ARTICLE 10) ---
print("\n🔎 Running Gender Bias Check...")
sex_counts = df['sex'].value_counts(normalize=True)
female_ratio = sex_counts.get('Female', 0)
male_ratio = sex_counts.get('Male', 0)

print(f"   Male Ratio: {male_ratio*100:.1f}% | Female Ratio: {female_ratio*100:.1f}%")

is_biased = female_ratio < BIAS_THRESHOLD
if is_biased:
    print(f"⚠️  WARNING: Female representation ({female_ratio*100:.1f}%) is below the {BIAS_THRESHOLD*100}% threshold.")
else:
    print("✅ Gender ratio is within acceptable limits.")

# --- 3. TECHNICAL INTEGRITY CHECK ---
print("\n🛡️  Running Deepchecks Data Integrity Suite...")
ds = Dataset(df, label='income', cat_features=['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country'])

integrity_suite = data_integrity()
result = integrity_suite.run(ds)
is_deepchecks_passed = result.passed()

# --- 4. REPORT GENERATION (Browser-Friendly) ---
os.makedirs("reports", exist_ok=True)
result.save_as_html(REPORT_PATH, as_widget=False, auto_open=False)
print(f"✅ HTML Report Generated at: {REPORT_PATH}")

# --- 5. GATE DECISION ---
print("\n" + "="*50)
print("             CI/CD GOVERNANCE GATE DECISION")
print("="*50)

if is_biased or not is_deepchecks_passed:
    print("⛔ GATE REJECTED: System violates one or more compliance standards.")
    if is_biased:
        print("   - Reason: Unacceptable Bias Risk (Low Female Representation)")
    if not is_deepchecks_passed:
        print("   - Reason: Data Integrity Failures (Conflicting Labels, etc.)")
    print("   -> Deployment Blocked.")
    sys.exit(1) # Block the CI/CD pipeline
else:
    print("✅ GATE APPROVED: Data is compliant and reliable.")
    print("   -> Proceeding to Model Training/Deployment.")
    sys.exit(0)
    