"""
===========================================================
AI-Based Intrusion Detection Framework
Adaptive Risk Scoring Module
===========================================================
"""

import numpy as np

from recommendation import (
    attack_weights,
    get_priority,
    get_severity
)


def calculate_risk(predicted_attack, probabilities):
    """
    Calculate Adaptive Risk Score

    Parameters
    ----------
    predicted_attack : str
        Predicted attack class

    probabilities : ndarray
        Output of model.predict_proba()

    Returns
    -------
    Dictionary containing complete risk assessment
    """

    # Highest probability
    confidence = float(np.max(probabilities)) * 100

    # Sort probabilities
    sorted_probs = np.sort(probabilities)

    highest = sorted_probs[-1] * 100
    second_highest = sorted_probs[-2] * 100

    # Prediction certainty
    certainty = highest - second_highest

    # Attack severity weight
    severity_weight = attack_weights[predicted_attack]

    # ----------------------------------------------------
    # Adaptive Risk Score
    # ----------------------------------------------------

    if predicted_attack == "BENIGN":

        risk_score = 5

    else:

        risk_score = (
            (0.5 * confidence) +
            (0.3 * severity_weight) +
            (0.2 * certainty)
        )

    risk_score = round(risk_score)

    if risk_score > 100:
        risk_score = 100

    # Priority
    priority = get_priority(risk_score)

    # Severity Level
    severity = get_severity(severity_weight)

    return {

        "attack": predicted_attack,

        "confidence": round(confidence, 2),

        "certainty": round(certainty, 2),

        "severity_weight": severity_weight,

        "severity": severity,

        "risk_score": risk_score,

        "priority": priority

    }