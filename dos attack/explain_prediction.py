"""
===========================================================
Explain Individual Predictions using TreeSHAP
===========================================================
"""

import joblib
import shap
import pandas as pd

# ==========================================================
# Feature Interpretations
# ==========================================================

feature_explanations = {

    "Flow Bytes/s": "High traffic volume strongly indicates flooding behaviour.",

    "Flow Packets/s": "Very high packet transmission rate suggests a DoS attack.",

    "Packet Length Mean": "Average packet size contributed to attack classification.",

    "Flow Duration": "Connection duration influenced the prediction.",

    "Average Packet Size": "Packet size characteristics affected the decision.",

    "Destination Port": "Destination port matched learned attack patterns.",

    "Total Fwd Packets": "Large number of forward packets indicates abnormal communication.",

    "Total Backward Packets": "Backward packet behaviour influenced the prediction.",

    "Total Length of Fwd Packets": "Forward traffic volume contributed to attack detection.",

    "Total Length of Bwd Packets": "Backward traffic volume affected the prediction.",

    "Flow IAT Mean": "Packet timing pattern differed from normal traffic.",

    "Flow IAT Std": "Variation in packet arrival times influenced the prediction.",

    "Active Mean": "Network activity duration affected the attack score.",

    "Idle Mean": "Idle periods helped distinguish attack traffic."
}

# ==========================================================
# Load Model
# ==========================================================

model = joblib.load("models/best_xgboost.pkl")

explainer = shap.TreeExplainer(model)


def explain_prediction(sample, feature_names, class_name, top_n=5):

    # Calculate SHAP values
    shap_values = explainer.shap_values(sample)

    # Handle multiclass models
    if isinstance(shap_values, list):

        class_index = int(model.predict(sample)[0])

        values = shap_values[class_index][0]

    else:

        prediction = int(model.predict(sample)[0])

        values = shap_values[0, :, prediction]

    explanation = pd.DataFrame({

        "Feature": feature_names,

        "SHAP Value": values

    })

    explanation["Impact"] = explanation["SHAP Value"].abs()

    explanation = explanation.sort_values(

        by="Impact",

        ascending=False

    )

    print("\n")
    print("=" * 70)
    print("TreeSHAP Explanation")
    print("=" * 70)

    for _, row in explanation.head(top_n).iterrows():

        feature = row["Feature"]

        contribution = row["SHAP Value"]

        # Handle column names with spaces
        feature_lookup = feature

        if feature_lookup not in sample.columns:
            feature_lookup = " " + feature

        value = sample.iloc[0][feature_lookup]

        interpretation = feature_explanations.get(

            feature.strip(),

            "This feature influenced the prediction."

        )

        print("\n--------------------------------------------------")

        print(f"Feature            : {feature.strip()}")

        print(f"Observed Value     : {value}")

        print(f"SHAP Contribution  : {contribution:.5f}")

        print(f"Interpretation     : {interpretation}")

    print("-" * 70)