import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# ===========================
# Load Dataset
# ===========================

print("="*60)
print("Loading Dataset...")
print("="*60)

df = pd.read_csv("sampled_dos_dataset.csv")

X = df.drop("Label", axis=1)
y = df["Label"]

X = X.replace([np.inf, -np.inf], 0)

# ===========================
# Train Test Split
# ===========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ===========================
# Feature Scaling
# ===========================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ===========================
# Models
# ===========================

models = {

    "Decision Tree":
        DecisionTreeClassifier(random_state=42),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),

    "XGBoost":
        XGBClassifier(
            objective="multi:softprob",
            num_class=6,
            eval_metric="mlogloss",
            random_state=42,
            tree_method="hist",
            n_estimators=100,
            max_depth=6
        ),

    "LightGBM":
        LGBMClassifier(
            random_state=42
        ),

    "CatBoost":
        CatBoostClassifier(
            verbose=0,
            random_state=42
        )
}

# ===========================
# Evaluation
# ===========================

results = []

print("\nTraining Models...\n")

for name, model in models.items():

    print(f"Training {name}...")

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        pred,
        average="weighted"
    )

    results.append([
        name,
        accuracy,
        precision,
        recall,
        f1
    ])

# ===========================
# Results
# ===========================

results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]
)

results_df = results_df.sort_values(
    by="Accuracy",
    ascending=False
)

print("\n")
print("="*70)
print(results_df)
print("="*70)

results_df.to_csv(
    "model_comparison_results.csv",
    index=False
)

print("\nResults saved successfully!")