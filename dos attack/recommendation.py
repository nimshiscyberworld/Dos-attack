"""
===========================================================
AI-Based Intrusion Detection Framework
Recommendation Module
===========================================================
"""

# Attack Severity Weight
attack_weights = {

    "BENIGN": 0,

    "DDoS": 100,

    "DoS Hulk": 90,

    "DoS GoldenEye": 80,

    "DoS slowloris": 70,

    "DoS Slowhttptest": 60

}


# Risk Priority
def get_priority(risk_score):

    if risk_score >= 90:
        return "Critical"

    elif risk_score >= 75:
        return "High"

    elif risk_score >= 50:
        return "Medium"

    else:
        return "Low"


# Severity Level
def get_severity(weight):

    if weight >= 90:
        return "Critical"

    elif weight >= 80:
        return "Very High"

    elif weight >= 70:
        return "High"

    elif weight >= 60:
        return "Medium"

    else:
        return "Low"


# Recommended Actions
recommendations = {

    "BENIGN": [

        "No malicious activity detected.",
        "Continue normal network monitoring.",
        "No immediate action required.",
        "Maintain firewall and IDS monitoring.",
        "Archive logs for future auditing."

    ],

    "DDoS": [

        "Enable DDoS Protection.",
        "Block malicious IP addresses.",
        "Enable rate limiting.",
        "Notify SOC immediately.",
        "Capture packets for forensic investigation."

    ],

    "DoS Hulk": [

        "Enable HTTP rate limiting.",
        "Block suspicious source IP.",
        "Deploy Web Application Firewall (WAF).",
        "Monitor CPU and memory utilization.",
        "Inspect abnormal HTTP requests."

    ],

    "DoS GoldenEye": [

        "Monitor HTTP connection usage.",
        "Enforce maximum concurrent connection limits.",
        "Enable HTTP connection throttling.",
        "Inspect abnormal HTTP sessions.",
        "Deploy Web Application Firewall (WAF)."

    ],

    "DoS slowloris": [

        "Reduce HTTP Keep-Alive timeout.",
        "Limit concurrent HTTP connections.",
        "Close incomplete HTTP sessions.",
        "Enable Slowloris protection module.",
        "Monitor long-lived HTTP connections."

    ],

    "DoS Slowhttptest": [

        "Reduce HTTP request timeout.",
        "Monitor slow HTTP requests.",
        "Limit idle HTTP connections.",
        "Configure request rate thresholds.",
        "Inspect suspicious client sessions."

    ]

}