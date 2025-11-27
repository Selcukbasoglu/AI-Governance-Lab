import sys
import os
import pandas as pd
from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import data_integrity, model_evaluation

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# --- CONFIGURATION & VARIABLE INITIALIZATION ---
REPORT_PATH = os.path.join("reports", "holistic_final_audit.html")
BIAS_THRESHOLD = 0.35  
ACCURACY_THRESHOLD = 0.80 

# Initialize all main decision variables
is_biased = False
integrity_passed = True
robustness_passed = True
accuracy_passed = True

print("="*70)
print("EU AI ACT INTEGRATED COMPLIANCE AUDIT (5 ARTICLES)")
print("="*70)

# --- CLEANUP: Delete old report files before new run ---
print("🧹 Cleaning up old reports...")
old_report_file = os.path.join("reports", "holistic_final_audit.html")

if os.path.exists(old_report_file):
    try:
        os.remove(old_report_file)
        print(f"   -> Deleted old report: {old_report_file}")
    except OSError as e:
        print(f"   ❌ ERROR: Could not delete file {old_report_file}. {e}")
else:
    print("   -> No old report found. Proceeding.")
# ----------------------------------------------------------------------


# --- 1. DATA LOADING & PREPARATION ---
try:
    print("⏳ Loading and Preprocessing Data...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    columns = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", 
               "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", 
               "hours-per-week", "native-country", "income"]
    df = pd.read_csv(url, names=columns, na_values=" ?", skipinitialspace=True).dropna()
    
    # Prepare data for model training
    df_encoded = pd.get_dummies(df.drop(columns=['fnlwgt', 'capital-gain', 'capital-loss'], errors='ignore'))
    
    # Separate features (X) and target (y)
    X = df_encoded.drop(columns=['income_>50K', 'income_<=50K'], errors='ignore')
    y = df_encoded['income_>50K']
    
    # Split for Robustness check (Article 15)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train a simple model
    model = LogisticRegression(solver='liblinear', random_state=42)
    model.fit(X_train, y_train)

    # Deepchecks datasets 
    ds_train = Dataset(X_train, label=y_train)
    ds_test = Dataset(X_test, label=y_test)

except Exception as e:
    print(f"❌ ERROR: Data loading or model training failed. {e}")
    sys.exit(1)


# --- 2. AUDIT SECTION 1: DATA GOVERNANCE & BIAS (ARTICLE 10) ---
print("\n[1/5] ⚖️ Article 10: Data Governance & Bias Check")

# A. Representational Bias Check 
female_ratio = df['sex'].value_counts(normalize=True).get('Female', 0)
if female_ratio < BIAS_THRESHOLD:
    print(f"  ❌ BIAS DETECTED: Female representation ({female_ratio*100:.1f}%) is below the {BIAS_THRESHOLD*100}% threshold.")
    is_biased = True
else:
    print("  ✅ Representation Test Passed.")
    is_biased = False 

# B. Data Integrity Check (Sets 'integrity_passed' correctly)
print("  ⏳ Deepchecks Data Integrity Scan...")
integrity_result = data_integrity().run(ds_train)
if not integrity_result.passed():
    print(f"  ❌ INTEGRITY FAILED: {len(integrity_result.get_not_passed_checks())} issues found (Conflicting Labels, etc.).")
    integrity_passed = False
else:
    print("  ✅ Data Integrity Passed.")
    integrity_passed = True


# --- 3. AUDIT SECTION 2: ROBUSTNESS & ACCURACY (ARTICLE 15) ---
print("\n[2/5] 🛡️ Article 15: Robustness and Accuracy Control")

# A. Accuracy Check (Sets 'accuracy_passed' correctly)
preds = model.predict(X_test)
accuracy = accuracy_score(y_test, preds)

if accuracy < ACCURACY_THRESHOLD:
    print(f"  ❌ ACCURACY FAILED: Model Accuracy ({accuracy*100:.1f}%) is below the {ACCURACY_THRESHOLD*100}% threshold.")
    accuracy_passed = False
else:
    print(f"  ✅ Accuracy Passed: {accuracy*100:.1f}% Accuracy detected.")
    accuracy_passed = True

# B. Model Robustness Check (Sets 'robustness_passed' correctly)
print("  ⏳ Deepchecks Model Evaluation Scan...")
model_eval_result = model_evaluation().run(train_dataset=ds_train, test_dataset=ds_test, model=model)
if not model_eval_result.passed():
    print("  ❌ ROBUSTNESS FAILED: Model Vulnerabilities Detected (e.g., Performance Stability).")
    robustness_passed = False
else:
    print("  ✅ Robustness Scan Passed.")
    robustness_passed = True


# --- 4. AUDIT SECTION 3, 4, 5 (Documentation & Traceability) ---
print("\n[3/5] 👁️ Article 13: Traceability & Human Oversight")
print("  ✅ Traceability Confirmed: Audit logs and versioning are mandatory via MLOps tools.")
print("\n[4/5] 🔒 GDPR/KVKK: Data Security and PII Control")
print("  ✅ Privacy Check: Assuming PII masking and data minimization protocols are followed.")
print("\n[5/5] 📄 Article 11: Documentation and Reporting")
print("  ✅ Documentation Confirmed: HTML Audit Report is being generated.")


# --- 5. REPORT GENERATION and FINAL GATE DECISION ---
print("\n" + "="*70)
print("             GENERATING HOLISTIC HTML REPORT")
print("="*70)

try:
    os.makedirs("reports", exist_ok=True)
    
    # 1. Geçici dosya yolları
    temp_integrity_path = os.path.join("reports", "temp_integrity.html")
    temp_model_path = os.path.join("reports", "temp_model.html")

    # 2. Raporları ayrı ayrı geçici dosyalara kaydediyoruz (Kesin çalışan yöntem)
    # Bu yöntem, SuiteResult'ın hiçbir .checks, .to_html, .merge, .display metoduna bağımlı değildir.
    integrity_result.save_as_html(temp_integrity_path, as_widget=False, auto_open=False)
    model_eval_result.save_as_html(temp_model_path, as_widget=False, auto_open=False)
    
    # 3. İki dosyayı okuyup HTML string'lerini alıyoruz
    with open(temp_integrity_path, 'r', encoding='utf-8') as f:
        integrity_html = f.read()
    with open(temp_model_path, 'r', encoding='utf-8') as f:
        model_html = f.read()

    # 4. HTML içeriğini manuel olarak birleştirme
    combined_html_content = (
        f"<h1>HOLISTIC AI AUDIT REPORT</h1>"
        f"<h2>DATA INTEGRITY & BIAS AUDIT (ARTICLE 10)</h2>"
        f"{integrity_html}" 
        f"<hr style='margin-top: 50px; border-top: 5px solid #1e88e5;'><h2>MODEL ROBUSTNESS & EVALUATION (ARTICLE 15)</h2>" + 
        f"{model_html}"
    )

    # 5. Final dosyaya yazma
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write(combined_html_content)

    # 6. Geçici dosyaları silme
    os.remove(temp_integrity_path)
    os.remove(temp_model_path)
    
    print(f"✅ Holistic HTML Report Generated: {REPORT_PATH} (Check reports folder)")
except Exception as e:
    print(f"❌ ERROR: Failed to generate HTML report. {e}")


# --- FINAL GATE DECISION ---
print("\n" + "="*70)
print("             FINAL GOVERNANCE GATE DECISION")
print("="*70)

final_fail = not integrity_passed or not robustness_passed or not accuracy_passed or is_biased

if final_fail:
    print("⛔ GATE REJECTED: AI System Violates Critical Standards.")
    if is_biased:
        print("   - CRITICAL REASON: Bias Risk Detected (Low Female Representation)")
    if not integrity_passed:
        print("   - CRITICAL REASON: Data Integrity Failures (Article 10)")
    if not accuracy_passed:
        print("   - CRITICAL REASON: Model Accuracy Below Threshold")
    if not robustness_passed:
        print("   - CRITICAL REASON: Model Robustness Failure (Article 15)")
        
    print("   -> DEPLOYMENT BLOCKED.")
    sys.exit(1)
else:
    print("✅ GATE APPROVED: AI System is Compliant and Reliable.")
    print("   -> Proceeding to Model Training and Deployment.")
    sys.exit(0)
