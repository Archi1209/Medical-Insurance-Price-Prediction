#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# Try seaborn (nice styling); fall back if missing
try:
    import seaborn as sns
    sns.set_theme(style="whitegrid")
    HAS_SNS = True
except Exception:
    HAS_SNS = False

# ---------- Optional XGBoost ----------
try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# ---------- Config ----------
RANDOM_STATE = 42
TEST_SIZE = 0.2

# ---------- Paths ----------
def script_dir():
    return os.path.dirname(os.path.abspath(__file__))

def outputs_dir():
    p = os.path.join(script_dir(), "outputs")
    os.makedirs(p, exist_ok=True)
    return p

def find_csv():
    # Try Desktop first, then same folder
    candidates = [
        os.path.join(os.path.expanduser("~"), "OneDrive", "Desktop", "insurance.csv"),
        os.path.join(os.path.expanduser("~"), "Desktop", "insurance.csv"),
        os.path.join(script_dir(), "insurance.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "Could not find 'insurance.csv'. Put it on Desktop or next to this script.\n"
        "Required columns: age, sex, bmi, children, smoker, region, charges"
    )

CSV_PATH = find_csv()
print(f"✅ Using dataset: {CSV_PATH}")

# ---------- Load ----------
df = pd.read_csv(CSV_PATH)
required = {"age","sex","bmi","children","smoker","region","charges"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Dataset missing columns: {missing}")

print("Rows:", len(df))
print(df.head().to_string(index=False))

# ---------- EDA GRAPHS ----------
out = outputs_dir()

def save_show(fig, filename):
    fig.savefig(os.path.join(out, filename), dpi=150, bbox_inches="tight")
    plt.show()

# 1) Distribution of charges
fig = plt.figure(figsize=(7,4))
if HAS_SNS:
    sns.histplot(df["charges"], bins=30, kde=True)
else:
    plt.hist(df["charges"], bins=30)
plt.title("Distribution of Insurance Charges")
plt.xlabel("Charges"); plt.ylabel("Frequency")
plt.tight_layout()
save_show(fig, "01_dist_charges.png")

# 2) Age vs Charges by smoker
fig = plt.figure(figsize=(6.5,4.5))
if HAS_SNS:
    sns.scatterplot(data=df, x="age", y="charges", hue="smoker", alpha=0.7)
else:
    colors = np.where(df["smoker"]=="yes", "tab:red", "tab:blue")
    plt.scatter(df["age"], df["charges"], c=colors, alpha=0.7)
    plt.legend(handles=[], title="Smoker (red=yes, blue=no)")
plt.title("Age vs Charges by Smoker")
plt.tight_layout()
save_show(fig, "02_scatter_age_charges_smoker.png")

# 3) Average charges by region
fig = plt.figure(figsize=(6.5,4.5))
if HAS_SNS:
    sns.barplot(data=df, x="region", y="charges", estimator=np.mean, ci=None)
else:
    means = df.groupby("region")["charges"].mean()
    plt.bar(means.index, means.values)
plt.title("Average Charges by Region")
plt.tight_layout()
save_show(fig, "03_bar_region_mean_charges.png")

# 4) Correlation heatmap for numerics
corr = df[["age","bmi","children","charges"]].corr()
fig = plt.figure(figsize=(4.8,4.2))
if HAS_SNS:
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
else:
    plt.imshow(corr, vmin=-1, vmax=1)
    plt.colorbar()
    plt.xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr)), corr.index)
plt.title("Correlation (Numerical Features)")
plt.tight_layout()
save_show(fig, "04_corr_heatmap.png")

# ---------- ML Pipeline ----------
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

X = df.drop(columns=["charges"])
y = df["charges"].values

numeric_features = ["age","bmi","children"]
categorical_features = ["sex","smoker","region"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=350, random_state=RANDOM_STATE, n_jobs=-1),
    # SVR is optional (can be slow on some machines). Uncomment to include.
    # "SVR(RBF)": SVR(kernel="rbf", C=100, gamma="scale", epsilon=0.1),
}
if HAS_XGB:
    models["XGBoost"] = xgb.XGBRegressor(
        n_estimators=600, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE,
        n_jobs=-1, tree_method="hist"
    )
else:
    print("[Info] XGBoost not installed; skipping. (Install: pip install xgboost)")

def metrics_safe(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)      # version-safe
    rmse = float(np.sqrt(mse))                    # RMSE via sqrt
    r2 = r2_score(y_true, y_pred)
    return {"MAE": float(mae), "MSE": float(mse), "RMSE": rmse, "R2": float(r2)}

results, pipes = {}, {}

for name, model in models.items():
    pipe = Pipeline([("pre", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    results[name] = metrics_safe(y_test, y_pred)
    pipes[name] = pipe
    print(f"{name}: {results[name]}")

# Save metrics
with open(os.path.join(out, "metrics.json"), "w") as f:
    json.dump(results, f, indent=2)

# ---------- Model comparison graphs ----------
names = list(results.keys())
mae_vals = [results[m]["MAE"] for m in names]
rmse_vals = [results[m]["RMSE"] for m in names]

fig = plt.figure(figsize=(6.8,4.2))
if HAS_SNS:
    sns.barplot(x=names, y=mae_vals)
else:
    plt.bar(names, mae_vals)
plt.ylabel("MAE"); plt.title("Model Comparison — MAE")
plt.tight_layout()
save_show(fig, "05_models_mae.png")

fig = plt.figure(figsize=(6.8,4.2))
if HAS_SNS:
    sns.barplot(x=names, y=rmse_vals)
else:
    plt.bar(names, rmse_vals)
plt.ylabel("RMSE"); plt.title("Model Comparison — RMSE")
plt.tight_layout()
save_show(fig, "06_models_rmse.png")

# ---------- Pred vs Actual for best model ----------
best_name = min(results, key=lambda k: results[k]["MAE"])
best_pipe = pipes[best_name]
y_pred_best = best_pipe.predict(X_test)

fig = plt.figure(figsize=(6,5))
plt.scatter(y_test, y_pred_best, s=20, alpha=0.7)
lo, hi = float(min(y_test.min(), y_pred_best.min())), float(max(y_test.max(), y_pred_best.max()))
plt.plot([lo, hi], [lo, hi], 'r--', lw=2)
plt.xlabel("Actual Charges"); plt.ylabel("Predicted Charges")
plt.title(f"Predicted vs Actual — {best_name}")
plt.tight_layout()
save_show(fig, "07_pred_vs_actual.png")

# ---------- Feature importance plot (if RF/XGB best) ----------
def get_feature_names(pre: ColumnTransformer):
    # Build list of feature names after ColumnTransformer
    names = []
    for name, trans, cols in pre.transformers_:
        if name == "num":
            names += list(cols)
        elif name == "cat":
            try:
                ohe = trans
                names += list(ohe.get_feature_names_out(cols))
            except Exception:
                # Fallback: generic names
                for c in cols:
                    names += [f"{c}_{i}" for i in range(1,6)]
    return names

final_model = best_pipe.named_steps["model"]
feature_names = get_feature_names(best_pipe.named_steps["pre"])

if hasattr(final_model, "feature_importances_"):
    importances = final_model.feature_importances_
    # guard length mismatch
    n = min(len(importances), len(feature_names))
    fi = pd.DataFrame({"feature": feature_names[:n], "importance": importances[:n]})
    fi = fi.sort_values("importance", ascending=False).head(15)
    fig = plt.figure(figsize=(7,5))
    if HAS_SNS:
        sns.barplot(y="feature", x="importance", data=fi)
    else:
        plt.barh(fi["feature"], fi["importance"])
        plt.gca().invert_yaxis()
    plt.title(f"Top Feature Importances — {best_name}")
    plt.tight_layout()
    save_show(fig, "08_feature_importances.png")
else:
    print(f"[Info] {best_name} does not expose 'feature_importances_' — skipping that plot.")

print("\nAll figures saved to:", out)
print("Done ✅")
