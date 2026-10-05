"""
======================================================================
MODEL 2 FINAL (CORRECTED): STABILIZER-STRUCTURE ONLY
======================================================================
تغییرات:
1. حذف کامل non_commuting_pairs
2. 8 ویژگی stabilizer
3. GroupKFold با graph_id
======================================================================
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

INPUT_FILE = "ml_full_dataset.csv"
RANDOM_STATE = 42
N_SPLITS = 5
TARGET = "P_L"
GROUP = "graph_id"

STABILIZER_FEATURES = [
    "mean_stab_weight", "var_stab_weight",
    "min_stab_weight", "max_stab_weight",
    "stab_weight_2", "stab_weight_3",
    "stab_weight_4", "stab_weight_5",
]


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


print("=" * 70)
print("MODEL 2 FINAL (CORRECTED): STABILIZER-STRUCTURE ONLY")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)
print(f"\n📂 Rows: {len(df):,}, Graphs: {df[GROUP].nunique():,}")

stab_list = [extract_stabilizer_features(row.get("stabilizer_generators")) for _, row in df.iterrows()]
for col in STABILIZER_FEATURES:
    df[col] = [f[col] for f in stab_list]

df_ml = df.dropna(subset=STABILIZER_FEATURES + [TARGET, GROUP]).copy()

X = df_ml[STABILIZER_FEATURES].to_numpy(dtype=float)
y = df_ml[TARGET].to_numpy(dtype=float)
groups = df_ml[GROUP].to_numpy()
y_log = np.log10(y)

print(f"\n📊 Features: {len(STABILIZER_FEATURES)} (stabilizer only)")

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
}

gkf = GroupKFold(n_splits=N_SPLITS)
print("\n" + "=" * 70)
print("GRAPH-DISJOINT 5-FOLD CV")
print("=" * 70)

results = {}
for model_name, model in models.items():
    print(f"\n🧠 {model_name}")
    print("-" * 50)
    fold_r2 = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y_log, groups), start=1):
        model.fit(X[train_idx], y_log[train_idx])
        y_pred = 10 ** model.predict(X[test_idx])
        r2 = r2_score(y[test_idx], y_pred)
        fold_r2.append(r2)
        print(f"   Fold {fold}: R² = {r2:.6f}")

    mean_r2 = np.mean(fold_r2)
    std_r2 = np.std(fold_r2, ddof=1)
    results[model_name] = {"mean_r2": mean_r2, "std_r2": std_r2}
    print(f"   📊 Mean R² = {mean_r2:.6f} ± {std_r2:.6f}")

print("\n" + "=" * 70)
print("FINAL RESULTS - MODEL 2")
print("=" * 70)
for name, res in results.items():
    print(f"   {name:25s}: R² = {res['mean_r2']:.6f} ± {res['std_r2']:.6f}")

print("\n✅ ANALYSIS COMPLETE")