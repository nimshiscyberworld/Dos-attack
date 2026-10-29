import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

from imblearn.over_sampling import SMOTE

from xgboost import XGBClassifier

print("="*60)
print("Loading Cleaned Dataset...")
print("="*60)

# Load dataset
df = pd.read_csv("sampled_dos_dataset.csv")

print("Dataset Shape:", df.shape)

# Features and Target
X = df.drop("Label", axis=1)
y = df["Label"]

print("\nTarget Distribution:")
print(y.value_counts())

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Shape :", X_train.shape)
print("Testing Shape  :", X_test.shape)

# Replace Inf values
X_train = X_train.replace([float("inf"), float("-inf")], 0)
X_test = X_test.replace([float("inf"), float("-inf")], 0)

# Scale Features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Apply SMOTE
print("\nApplying SMOTE...")

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

print("Balanced Training Shape:", X_train_smote.shape)

print("\nBalanced Class Distribution:")
print(pd.Series(y_train_smote).value_counts())

# Build Model
print("\nTraining XGBoost...")

model = XGBClassifier(
    objective="multi:softmax",
    num_class=6,
    n_estimators=200,
    learning_rate=0.1,
    max_depth=8,
    random_state=42,
    eval_metric="mlogloss"
)

model.fit(X_train_smote, y_train_smote)

print("\nTraining Completed!")

# Prediction
pred = model.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, pred)

print("\nAccuracy :", round(acc*100,2), "%")

print("\nClassification Report\n")

print(classification_report(y_test, pred))

# Save Model
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/xgboost_model.pkl")

joblib.dump(scaler, "models/scaler.pkl")

print("\nModel Saved Successfully!")