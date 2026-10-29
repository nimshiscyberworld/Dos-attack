import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

# ===============================
# Load Dataset
# ===============================

print("Loading Dataset...")

df = pd.read_csv("sampled_dos_dataset.csv")

X = df.drop("Label", axis=1)
y = df["Label"]

X = X.replace([np.inf, -np.inf], 0)

# ===============================
# Train Test Split
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ===============================
# Load Model
# ===============================

print("Loading Model...")

model = joblib.load("models/best_xgboost.pkl")

# ===============================
# Predict Probabilities
# ===============================

y_score = model.predict_proba(X_test)

classes = np.unique(y)

y_test_bin = label_binarize(y_test, classes=classes)

# ===============================
# ROC Curve
# ===============================

plt.figure(figsize=(8,6))

for i in range(len(classes)):

    fpr, tpr, _ = roc_curve(
        y_test_bin[:, i],
        y_score[:, i]
    )

    roc_auc = auc(fpr, tpr)

    plt.plot(
        fpr,
        tpr,
        label=f"Class {i} (AUC={roc_auc:.4f})"
    )

plt.plot([0,1],[0,1],'k--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("Multi-Class ROC Curve")

plt.legend()

plt.tight_layout()

plt.savefig("roc_curve.png",dpi=300)

plt.show()

print("ROC Curve Saved Successfully!")