import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

print("="*60)
print("Loading Dataset...")
print("="*60)

df = pd.read_csv("sampled_dos_dataset.csv")

X = df.drop("Label", axis=1)
y = df["Label"]

X = X.replace([np.inf,-np.inf],0)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("Loading Optimized Model...")

model = joblib.load("models/best_xgboost.pkl")

print("Predicting...")

pred = model.predict(X_test)

print("\nClassification Report\n")
print(classification_report(y_test,pred))

# ==========================
# Confusion Matrix
# ==========================

cm = confusion_matrix(y_test,pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot(cmap="Blues")

plt.title("Confusion Matrix")

plt.savefig("confusion_matrix.png",dpi=300)

plt.show()

# ==========================
# Feature Importance
# ==========================

importance = model.feature_importances_

feature_names = X.columns

indices = np.argsort(importance)[::-1][:20]

plt.figure(figsize=(10,7))

plt.barh(
    range(len(indices)),
    importance[indices]
)

plt.yticks(
    range(len(indices)),
    feature_names[indices]
)

plt.gca().invert_yaxis()

plt.title("Top 20 Important Features")

plt.tight_layout()

plt.savefig("feature_importance.png",dpi=300)

plt.show()

print("\nEvaluation Completed Successfully!")