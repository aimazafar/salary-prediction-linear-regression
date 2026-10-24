# Project 2/3: Salary Prediction (Multiple Linear Regression)
# Run: pip install pandas numpy scikit-learn joblib
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

CSV = "salary_data.csv"

if not os.path.exists(CSV):
    rng = np.random.default_rng(7)
    n = 500
    df = pd.DataFrame({
        "experience": rng.uniform(0, 15, n).round(1),
        "test_score": rng.integers(40, 100, n),
        "department": rng.choice(["IT", "HR", "Sales"], n),
    })
    bonus = df.department.map({"IT": 15000, "HR": 0, "Sales": 7000})
    df["salary"] = (30000 + df.experience*4000 + df.test_score*300 + bonus
                    + rng.normal(0, 4000, n)).round(0)
    df.to_csv(CSV, index=False)

df = pd.read_csv(CSV)
print(df.head(), "\n")

# Categorical -> numbers (one-hot)
df = pd.get_dummies(df, columns=["department"], drop_first=True)

def run(features, label):
    X, y = df[features], df["salary"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    m = LinearRegression().fit(Xtr, ytr)
    p = m.predict(Xte)
    rmse, r2 = np.sqrt(mean_squared_error(yte, p)), r2_score(yte, p)
    print(f"{label:28s} RMSE: {rmse:10,.2f}   R2: {r2:.4f}")
    return m, r2, features

# Single vs multiple feature comparison
single = run(["experience"], "Single (experience)")
multi_cols = [c for c in df.columns if c != "salary"]
multi = run(multi_cols, "Multiple (all features)")

best = max([single, multi], key=lambda t: t[1])
joblib.dump({"model": best[0], "features": best[2]}, "salary_model.pkl")
print("\nBest model saved -> salary_model.pkl  (features:", best[2], ")")

# Prediction example
saved = joblib.load("salary_model.pkl")
new = pd.DataFrame([{"experience": 5, "test_score": 80, "department_IT": 1, "department_Sales": 0}])
print("Predicted salary (5 yrs, score 80, IT): {:,.0f}".format(saved["model"].predict(new[saved["features"]])[0]))
