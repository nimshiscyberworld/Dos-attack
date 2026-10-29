"""
===========================================================
AI-Based Risk-Aware Intrusion Detection Framework
Prediction Module
===========================================================
"""

import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from risk_scoring import calculate_risk
from recommendation import recommendations
from explain_shap import explain_prediction

# ==========================================================
# Load Dataset
# ==========================================================

print("=" * 80)
print("Loading Dataset...")
print("=" * 80)

df = pd.read_csv("sampled_dos_dataset.csv")

X = df.drop("Label", axis=1)
y = df["Label"]

# Replace infinite values
X = X.replace([np.inf, -np.inf], 0)

# ==========================================================
# Train-Test Split
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================================================
# Load Optimized XGBoost Model
# ==========================================================

print("\nLoading Optimized XGBoost Model...\n")

model = joblib.load("models/best_xgboost.pkl")

# ==========================================================
# Label Mapping
# ==========================================================

label_names = {
    0: "BENIGN",
    1: "DDoS",
    2: "DoS GoldenEye",
    3: "DoS Hulk",
    4: "DoS Slowhttptest",
    5: "DoS slowloris"
}

# ==========================================================
# Select Samples
# ==========================================================

samples = X_test.iloc[:20]

predictions = model.predict(samples)

probabilities = model.predict_proba(samples)

# ==========================================================
# AI Risk Assessment Report
# ==========================================================

print("=" * 100)
print("         AI-BASED RISK-AWARE INTRUSION DETECTION REPORT")
print("=" * 100)

for i in range(len(samples)):

    predicted_attack = label_names[int(predictions[i])]

    probability = probabilities[i]

    # ------------------------------------------
    # Calculate Risk
    # ------------------------------------------

    result = calculate_risk(
        predicted_attack,
        probability
    )

    # ------------------------------------------
    # Display Prediction
    # ------------------------------------------

    print("\n")
    print("=" * 80)
    print(f"Sample {i+1}")
    print("=" * 80)

    print(f"Predicted Attack      : {result['attack']}")
    print(f"Confidence            : {result['confidence']} %")
    print(f"Prediction Certainty  : {result['certainty']} %")
    print(f"Severity Weight       : {result['severity_weight']}")
    print(f"Severity Level        : {result['severity']}")
    print(f"Risk Score            : {result['risk_score']} /100")
    print(f"Priority              : {result['priority']}")

    # ======================================================
    # SHAP Explanation
    # ======================================================

    explain_prediction(

        sample=samples.iloc[[i]],

        feature_names=X.columns.tolist(),

        class_name=result["attack"],

        top_n=5

    )

    # ======================================================
    # Recommendations
    # ======================================================

    print("\nRecommended Mitigation")
    print("-" * 60)

    for action in recommendations[result["attack"]]:
        print("✓", action)

print("\n")
print("=" * 100)
print("Assessment Completed Successfully")
print("=" * 100)