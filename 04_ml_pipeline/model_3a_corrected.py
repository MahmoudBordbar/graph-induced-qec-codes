"""
======================================================================
MODEL 3A: TOPOLOGY + STABILIZER (No Engineered Features)
======================================================================
هدف:
    جداسازی سهم واقعی ترکیب topology + stabilizer
    بدون هیچ ویژگی engineered

ویژگی‌ها:
    - 6 topology
    - 8 stabilizer (بدون non_commuting_pairs)
    ─────────────────────
    - 14 ویژگی کل

شرایط:
    - GroupKFold با graph_id
    - 5 fold
    - log10(P_L) transform (مشابه Model 3)
    - 22,568 ردیف، 728 گراف
======================================================================
"""

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
)
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

INPUT_FILE = "ml_full_dataset.csv"
RANDOM_STATE = 42
N_SPLITS = 5
TARGET = "P_L"
GROUP = "graph_id"

# ================================================================
# ویژگی‌ها
# ================================================================

TOPOLOGY_FEATURES = [
    "edges", "min_degree", "max_degree",
    "degree_variance", "diameter", "triangles",
]

STABILIZER_FEATURES = [
    "mean_stab_weight", "var_stab_weight",
    "min_stab_weight", "max_stab_weight",
    "stab_weight_2", "stab_weight_3",
    "stab_weight_4", "stab_weight_5",
]

ALL_FEATURES = TOPOLOGY_FEATURES + STABILIZER_FEATURES

# ================================================================
# توابع استخراج ویژگی‌های stabilizer
# ================================================================

def pauli_weight(p):
    if not isinstance(p, str):
        return 0
    return sum(1 for c in p if c != "I")


def parse_stabilizers(g):
    if pd.isna(g) or not isinstance(g, str):
        return []
    return g.split(";")


def extract_stabilizer_features(generators_str):
    generators = parse_stabilizers(generators_str)
    default = {
        "mean_stab_weight": 0.0, "var_stab_weight": 0.0,
        "min_stab_weight": 0, "max_stab_weight": 0,
        "stab_weight_2": 0, "stab_weight_3": 0,
        "stab_weight_4": 0, "stab_weight_5": 0,
    }
    if not generators:
        return default

    weights = [pauli_weight(g) for g in generators]
    wc = {2: 0, 3: 0, 4: 0, 5: 0}
    for w in weights:
        if w in wc:
            wc[w] += 1

    return {
        "mean_stab_weight": float(np.mean(weights)) if weights else 0.0,
        "var_stab_weight": float(np.var(weights)) if len(weights) > 1 else 0.0,
        "min_stab_weight": int(np.min(weights)) if weights else 0,
        "max_stab_weight": int(np.max(weights)) if weights else 0,
        "stab_weight_2": wc[2],
        "stab_weight_3": wc[3],
        "stab_weight_4": wc[4],
        "stab_weight_5": wc[5],
    }


# ================================================================
# بارگذاری و پردازش
# ================================================================

print("=" * 70)
print("MODEL 3A: TOPOLOGY + STABILIZER (No Engineered)")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)
print(f"\n📂 Rows: {len(df):,}")
print(f"   Graphs: {df[GROUP].nunique():,}")

print("\n🔧 Extracting stabilizer features (8 features)...")
stab_list = [
    extract_stabilizer_features(row.get("stabilizer_generators"))
    for _, row in df.iterrows()
]
for col in STABILIZER_FEATURES:
    df[col] = [f[col] for f in stab_list]

print(f"\n📊 Feature breakdown:")
print(f"   Topology:    {len(TOPOLOGY_FEATURES)}")
print(f"   Stabilizer:  {len(STABILIZER_FEATURES)}")
print(f"   ─────────────────────")
print(f"   Total:       {len(ALL_FEATURES)}")

# ================================================================
# آماده‌سازی داده
# ================================================================

required_cols = ALL_FEATURES + [TARGET, GROUP]
df_ml = df.dropna(subset=required_cols).copy()

X = df_ml[ALL_FEATURES].to_numpy(dtype=float)
y = df_ml[TARGET].to_numpy(dtype=float)
groups = df_ml[GROUP].to_numpy()

y_log = np.log10(y)

print(f"\n📊 Final dataset: {len(df_ml):,} rows, {len(ALL_FEATURES)} features")

# ================================================================
# مدل‌ها
# ================================================================

models = {
    "Random Forest": RandomForestRegressor(
        n_estimators=500, max_depth=14, min_samples_split=3,
        random_state=RANDOM_STATE, n_jobs=-1
    ),
    "Extra Trees": ExtraTreesRegressor(
        n_estimators=500, max_depth=16, min_samples_split=2,
        random_state=RANDOM_STATE, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, random_state=RANDOM_STATE
    ),
    "XGBoost (LogScale)": XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_STATE, n_jobs=-1
    ),
}

# ================================================================
# Graph-Disjoint 5-Fold CV
# ================================================================

print("\n" + "=" * 70)
print("GRAPH-DISJOINT 5-FOLD CV WITH LOG-TRANSFORM")
print("=" * 70)

gkf = GroupKFold(n_splits=N_SPLITS)
results = {}

for model_name, model in models.items():
    print(f"\n🧠 {model_name}")
    print("-" * 50)
    fold_r2 = []
    fold_rmse = []
    fold_mae = []

    for fold, (train_idx, test_idx) in enumerate(
        gkf.split(X, y_log, groups), start=1
    ):
        model.fit(X[train_idx], y_log[train_idx])
        y_pred_log = model.predict(X[test_idx])
        y_pred = 10 ** y_pred_log

        r2 = r2_score(y[test_idx], y_pred)
        rmse = np.sqrt(mean_squared_error(y[test_idx], y_pred))
        mae = mean_absolute_error(y[test_idx], y_pred)

        fold_r2.append(r2)
        fold_rmse.append(rmse)
        fold_mae.append(mae)

        print(f"   Fold {fold}: R²={r2:.6f}, RMSE={rmse:.6e}, MAE={mae:.6e}")

    mean_r2 = np.mean(fold_r2)
    std_r2 = np.std(fold_r2, ddof=1)
    results[model_name] = {
        "mean_r2": mean_r2,
        "std_r2": std_r2,
        "mean_rmse": np.mean(fold_rmse),
        "mean_mae": np.mean(fold_mae),
    }
    print(f"   📊 Mean R² = {mean_r2:.6f} ± {std_r2:.6f}")

# ================================================================
# اهمیت ویژگی‌ها
# ================================================================

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE (Random Forest, full data)")
print("=" * 70)

rf_full = RandomForestRegressor(
    n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1
)
rf_full.fit(X, y_log)

imp = pd.DataFrame({
    "feature": ALL_FEATURES,
    "importance": rf_full.feature_importances_
}).sort_values("importance", ascending=False)

for _, r in imp.iterrows():
    print(f"   {r['feature']:25s}: {r['importance']:.6f}")

# ================================================================
# گزارش نهایی
# ================================================================

print("\n" + "=" * 70)
print("FINAL RESULTS - MODEL 3A")
print("=" * 70)

for name, res in results.items():
    print(f"   {name:25s}: R² = {res['mean_r2']:.6f} ± {res['std_r2']:.6f}")

best = max(results, key=lambda x: results[x]['mean_r2'])
print(f"\n🏆 Model with highest mean graph-disjoint R²: {best}")
print(f"   R² = {results[best]['mean_r2']:.6f} ± {results[best]['std_r2']:.6f}")

# ================================================================
# ذخیره
# ================================================================

imp.to_csv("model3a_feature_importance.csv", index=False)
pd.DataFrame(results).T.to_csv("model3a_summary.csv")

print("\n📁 Saved:")
print("   model3a_feature_importance.csv")
print("   model3a_summary.csv")
print("\n✅ ANALYSIS COMPLETE")