import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier

print("=" * 60)
print("Loading Dataset...")
print("=" * 60)

# Load cleaned dataset
df = pd.read_csv("cleaned_dos_dataset.csv")

# Features and Target
X = df.drop("Label", axis=1)
y = df["Label"]

# Replace infinity values
X = X.replace([np.inf, -np.inf], 0)

# Pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("xgb", XGBClassifier(
        objective="multi:softprob",
        num_class=6,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric="mlogloss",
        tree_method="hist"
    ))
])

# 5-Fold Stratified Cross Validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

print("\nRunning 5-Fold Cross Validation...\n")

scores = cross_val_score(
    pipeline,
    X,
    y,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1
)

print("Fold Accuracies:")
for i, score in enumerate(scores, start=1):
    print(f"Fold {i}: {score:.4f}")

print("\nMean Accuracy :", scores.mean())
print("Std Deviation :", scores.std())