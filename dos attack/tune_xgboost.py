import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier

print("="*60)
print("Loading Dataset...")
print("="*60)

df = pd.read_csv("sampled_dos_dataset.csv")

X = df.drop("Label", axis=1)
y = df["Label"]

X = X.replace([np.inf, -np.inf], 0)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

param_grid = {

    "n_estimators":[100,150,200],

    "max_depth":[4,6,8],

    "learning_rate":[0.01,0.05,0.1],

    "subsample":[0.8,0.9,1.0],

    "colsample_bytree":[0.8,0.9,1.0]
}

xgb = XGBClassifier(

    objective="multi:softprob",

    num_class=6,

    eval_metric="mlogloss",

    tree_method="hist",

    random_state=42
)

search = RandomizedSearchCV(

    estimator=xgb,

    param_distributions=param_grid,

    n_iter=10,

    cv=3,

    scoring="accuracy",

    verbose=2,

    random_state=42,

    n_jobs=-1

)

print("\nRunning Hyperparameter Search...\n")

search.fit(X_train,y_train)

print("\nBest Parameters")

print(search.best_params_)

print("\nBest CV Accuracy")

print(search.best_score_)

best_model = search.best_estimator_

pred = best_model.predict(X_test)

acc = accuracy_score(y_test,pred)

print("\nTest Accuracy:",acc)

joblib.dump(best_model,"models/best_xgboost.pkl")

print("\nOptimized model saved.")