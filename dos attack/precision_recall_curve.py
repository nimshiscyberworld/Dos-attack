import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve, average_precision_score

# Load dataset
df = pd.read_csv("sampled_dos_dataset.csv")

X = df.drop("Label", axis=1)
y = df["Label"]

X = X.replace([np.inf, -np.inf], 0)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Load model
model = joblib.load("models/best_xgboost.pkl")

# Predict probabilities
y_score = model.predict_proba(X_test)

classes = np.unique(y)

y_test_bin = label_binarize(y_test, classes=classes)

plt.figure(figsize=(8,6))

for i in range(len(classes)):

    precision, recall, _ = precision_recall_curve(
        y_test_bin[:, i],
        y_score[:, i]
    )

    ap = average_precision_score(
        y_test_bin[:, i],
        y_score[:, i]
    )

    plt.plot(
        recall,
        precision,
        label=f"Class {i} (AP={ap:.4f})"
    )

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Multi-Class Precision-Recall Curve")
plt.legend()

plt.tight_layout()

plt.savefig("precision_recall_curve.png", dpi=300)

plt.show()

print("Precision-Recall Curve Saved Successfully!")