"""
======================================================================
TEST 02: Model 2 (Stabilizer only) — Verify R² = 0.338 ± 0.026
======================================================================
هدف: تأیید R² مدل 2 با 8 ویژگی stabilizer (بدون non_commuting)
======================================================================
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
)
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score

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


def extract_stabilizer_features(generators_str):
    if pd.isna(generators_str) or not isinstance(generators_str, str):
        return {k: 0 for k in STABILIZER_FEATURES}

    generators = generators_str.split(";")
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


def main():
    print("=" * 70)
    print("TEST 02: Model 2 R² Verification")
    print("=" * 70)

    df = pd.read_csv(INPUT_FILE)
    print(f"\n📂 Rows: {len(df):,}, Graphs: {df[GROUP].nunique():,}")

    # استخراج ویژگیهای stabilizer
    features = [extract_stabilizer_features(row.get("stabilizer_generators"))
                for _, row in df.iterrows()]
    for col in STABILIZER_FEATURES:
        df[col] = [f[col] for f in features]

    df_ml = df.dropna(subset=STABILIZER_FEATURES + [TARGET, GROUP]).copy()

    X = df_ml[STABILIZER_FEATURES].to_numpy(dtype=float)
    y = df_ml[TARGET].to_numpy(dtype=float)
    groups = df_ml[GROUP].to_numpy()
    y_log = np.log10(y)

    print(f"📊 Final: {len(df_ml):,} rows, {len(STABILIZER_FEATURES)} features")

    # مدلها
    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=500, max_depth=14, min_samples_split=3,
            random_state=RANDOM_STATE, n_jobs=-1),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=500, max_depth=16, min_samples_split=2,
            random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, random_state=RANDOM_STATE),
    }

    gkf = GroupKFold(n_splits=N_SPLITS)
    results = {}

    for name, model in models.items():
        fold_r2 = []
        for fold, (tr, te) in enumerate(gkf.split(X, y_log, groups), start=1):
            model.fit(X[tr], y_log[tr])
            y_pred = 10 ** model.predict(X[te])
            fold_r2.append(r2_score(y[te], y_pred))

        mean_r2 = np.mean(fold_r2)
        std_r2 = np.std(fold_r2, ddof=1)
        results[name] = (mean_r2, std_r2)
        print(f"\n   {name:22s}: R² = {mean_r2:.6f} ± {std_r2:.6f}")

    print("\n" + "=" * 70)
    print("COMPARISON WITH ARTICLE")
    print("=" * 70)

    # ادعای مقاله
    article_claim = {
        "Random Forest": 0.337718,
        "Extra Trees": 0.338156,
        "Gradient Boosting": 0.338120,
    }
    article_std = 0.026

    print(f"\n{'Model':22s} | {'Article':>10s} | {'This run':>10s} | Match?")
    print("-" * 70)

    all_match = True
    for name, (r2, std) in results.items():
        claimed = article_claim.get(name, None)
        if claimed is None:
            continue
        match = abs(r2 - claimed) < 0.005
        if not match:
            all_match = False
        symbol = "✅" if match else "❌"
        print(f"{name:22s} | {claimed:10.6f} | {r2:10.6f} | {symbol}")

    print("\n" + "=" * 70)
    if all_match:
        print("🎉 TEST 02 PASSED — Article claim verified!")
    else:
        print("⚠️  TEST 02: Some values differ from article")
        print("    (Article reports R² ≈ 0.338 ± 0.026)")
    print("=" * 70)


if __name__ == "__main__":
    main()