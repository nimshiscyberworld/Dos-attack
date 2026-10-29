"""
===========================================================
TreeSHAP Explanation Module
Explains each individual prediction
===========================================================
"""

import joblib
import shap
import numpy as np
import pandas as pd


# ==========================================================
# Feature Interpretations
# ==========================================================

feature_explanations = {

    "Flow Bytes/s":
        "High traffic volume strongly indicates flooding behaviour.",

    " Flow Packets/s":
        "Very high packet transmission rate suggests a DoS attack.",

    " Packet Length Mean":
        "Average packet size contributed to attack classification.",

    " Flow Duration":
        "Connection duration influenced the prediction.",

    " Average Packet Size":
        "Packet size characteristics affected the decision.",

    " Destination Port":
        "Destination port matched learned attack patterns.",

    " Total Fwd Packets":
        "Large number of forward packets indicates abnormal communication.",

    " Total Backward Packets":
        "Backward packet behaviour influenced the prediction.",

    " Total Length of Fwd Packets":
        "Forward traffic volume contributed to attack detection.",

    " Total Length of Bwd Packets":
        "Backward traffic volume affected the prediction.",

    " Fwd Packet Length Mean":
        "Average forward packet length influenced the classifier.",

    " Bwd Packet Length Mean":
        "Average backward packet length contributed to the decision.",

    " Flow IAT Mean":
        "Packet timing pattern differed from normal traffic.",

    " Flow IAT Std":
        "Variation in packet arrival times influenced the prediction.",

    " Active Mean":
        "Network activity duration affected the attack score.",

    " Idle Mean":
        "Idle periods helped distinguish attack traffic.",

    " Init_Win_bytes_forward":
        "TCP window size influenced attack classification.",

    " Init_Win_bytes_backward":
        "Backward TCP window behaviour contributed to prediction.",

    " act_data_pkt_fwd":
        "Forward data packet count affected the decision.",

    " min_seg_size_forward":
        "TCP segment size influenced attack detection."
}

# ----------------------------------------------------------
# Load Optimized XGBoost Model
# ----------------------------------------------------------

model = joblib.load("models/best_xgboost.pkl")

# Create TreeSHAP Explainer
explainer = shap.TreeExplainer(model)


def explain_prediction(sample, feature_names, class_name, top_n=5):

    """
    Explain one prediction using TreeSHAP.
    """

    # ---------------------------------------------
    # Calculate SHAP values
    # ---------------------------------------------

    shap_values = explainer(sample)

    # ---------------------------------------------
    # Predicted class index
    # ---------------------------------------------

    pred_class = int(model.predict(sample)[0])

    values = shap_values.values

    # ---------------------------------------------
    # Handle different SHAP output formats
    # ---------------------------------------------

    if values.ndim == 3:
        # New SHAP versions
        sample_values = values[0, :, pred_class]

    elif values.ndim == 2:
        # Binary models
        sample_values = values[0]

    else:
        raise ValueError(
            f"Unsupported SHAP output shape: {values.shape}"
        )

    # ---------------------------------------------
    # Create explanation dataframe
    # ---------------------------------------------

    explanation = pd.DataFrame({

        "Feature": feature_names,

        "Contribution": sample_values

    })

    explanation["Impact"] = explanation["Contribution"].abs()

    explanation = explanation.sort_values(
        by="Impact",
        ascending=False
    )

    # ---------------------------------------------
    # Print explanation
    # ---------------------------------------------

    print("\n")
    print("Top Features Responsible")
    print("-" * 60)

    for _, row in explanation.head(top_n).iterrows():

        arrow = "↑" if row["Contribution"] > 0 else "↓"

        print(
            f"{row['Feature']:<35}"
            f"{arrow} "
            f"{row['Contribution']:.5f}"
        )

    print("\nExplanation")
    print("-" * 60)

    for _, row in explanation.head(top_n).iterrows():

        if row["Contribution"] > 0:

            print(
                f"✓ {row['Feature']} increased the probability of {class_name}."
            )

        else:

            print(
                f"✓ {row['Feature']} reduced the probability of {class_name}."
            )

    print("-" * 60)