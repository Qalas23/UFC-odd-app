import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import joblib
import os

# ===========================
# 📥 Load & Prepare Data
# ===========================

df = pd.read_csv("data/large_dataset.csv")

# Features for all modellen
features = [
    "age_diff", "height_diff", "weight_diff", "reach_diff",
    "SLpM_total_diff", "SApM_total_diff", "sig_str_acc_total_diff",
    "str_def_total_diff", "td_avg_diff", "td_acc_total_diff",
    "td_def_total_diff", "sub_avg_diff"
]

# Verwijder incomplete rijen
df = df.dropna(subset=features + ["winner", "method", "finish_round"])

# Feature matrix & labels
X = df[features]
y_win = (df["winner"] == "Red").astype(int)
y_method = df["method"]
y_round = df["finish_round"]

# Zorg dat models map bestaat
os.makedirs("models", exist_ok=True)

# ===========================
# 🧠 Train Win Models
# ===========================

print("🎯 Training win prediction models...")

rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y_win)
lr = LogisticRegression(max_iter=1000).fit(X, y_win)
xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss").fit(X, y_win)

joblib.dump(rf, "models/random_forest.pkl")
joblib.dump(lr, "models/log_reg.pkl")
joblib.dump(xgb, "models/xgb.pkl")

print("✅ Win-predictie modellen opgeslagen.")

# ===========================
# 🧠 Train Method & Round Models
# ===========================

print("📦 Training method & round prediction models...")

method_model = RandomForestClassifier(n_estimators=100, random_state=42)
method_model.fit(X, y_method)
joblib.dump(method_model, "models/method_model.pkl")

round_model = RandomForestClassifier(n_estimators=100, random_state=42)
round_model.fit(X, y_round)
joblib.dump(round_model, "models/round_model.pkl")

print("✅ Method & round modellen opgeslagen.")
