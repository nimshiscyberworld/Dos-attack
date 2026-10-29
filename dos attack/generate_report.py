"""
===========================================================
AI-Based Risk-Aware Intrusion Detection Framework
Report Generation Module
===========================================================
"""

import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from risk_scoring import calculate_risk
from recommendation import recommendations

# ==========================================================
# Load Dataset
# ==========================================================

print("Loading Dataset...")

df = pd.read_csv("sampled_dos_dataset.csv")

X = df.drop("Label", axis=1)
y = df["Label"]

X = X.replace([np.inf, -np.inf], 0)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================================================
# Load Model
# ==========================================================

print("Loading Optimized XGBoost Model...")

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
# Predict
# ==========================================================

samples = X_test.iloc[:100]

predictions = model.predict(samples)

probabilities = model.predict_proba(samples)

report = []

for i in range(len(samples)):

    attack = label_names[int(predictions[i])]

    result = calculate_risk(attack, probabilities[i])

    recommendation_text = " | ".join(recommendations[result["attack"]])

    report.append({

        "Sample": i + 1,

        "Predicted Attack": result["attack"],

        "Confidence (%)": result["confidence"],

        "Prediction Certainty": result["certainty"],

        "Severity Weight": result["severity_weight"],

        "Severity": result["severity"],

        "Risk Score": result["risk_score"],

        "Priority": result["priority"],

        "Recommendations": recommendation_text

    })

# ==========================================================
# Save CSV Report
# ==========================================================

report_df = pd.DataFrame(report)

report_df.to_csv("AI_Risk_Assessment_Report.csv", index=False)

print("\n========================================")
print("Report Generated Successfully!")
print("Saved as:")
print("AI_Risk_Assessment_Report.csv")
print("========================================")