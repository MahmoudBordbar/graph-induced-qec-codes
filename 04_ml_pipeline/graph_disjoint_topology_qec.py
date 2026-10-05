"""
======================================================================
Graph-Disjoint Topology-Only QEC Prediction
======================================================================

هدف:
    بررسی اینکه ویژگی‌های توپولوژیکی خالص گراف تا چه حد می‌توانند
    P_L را بدون استفاده از subgroup/QEC features پیش‌بینی کنند.

ویژگی‌های حذف‌شده:
    - distance
    - subgroup_id
    - N2, N3, N4, N5
    - A1
    - N2_X, N2_Y, N2_Z
    - stabilizer-related features
    - logical/QEC-derived features

نکته مهم:
    تمام subgroupهای یک graph_id در یک fold قرار می‌گیرند.
    بنابراین هیچ graph topology مشترکی بین train و test وجود ندارد.

تحلیل‌های اصلی:
    1. Graph-disjoint 5-fold CV
    2. Random Forest
    3. Gradient Boosting
    4. Within-graph variance
    5. Between-graph variance
    6. Total variance decomposition
    7. Graph-level theoretical R² ceiling
======================================================================
"""

import time
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)


# ======================================================================
# تنظیمات
# ======================================================================

INPUT_FILE = "ml_full_dataset.csv"

RANDOM_STATE = 42
N_SPLITS = 5


# ======================================================================
# ویژگی‌های توپولوژیکی مستقل‌تر
# ======================================================================

# برای گراف‌های connected با V=5:
#
# mean_degree = 2E/V = 2E/5
#
# cycle_rank = E - V + 1 = E - 4
#
# بنابراین mean_degree و cycle_rank اطلاعات مستقلی از edges ندارند.
#
# برای تحلیل تمیزتر، آنها را از مدل اصلی حذف می‌کنیم.

TOPOLOGY_FEATURES = [
    "edges",
    "min_degree",
    "max_degree",
    "degree_variance",
    "diameter",
    "triangles",
]

TARGET = "P_L"
GROUP = "graph_id"


# ======================================================================
# چاپ عنوان
# ======================================================================

print("=" * 78)
print("GRAPH-DISJOINT TOPOLOGY-ONLY QEC PREDICTION")
print("=" * 78)

print("\n📂 Loading dataset...")
df = pd.read_csv(INPUT_FILE)

print(f"   Rows       : {len(df):,}")
print(f"   Columns    : {len(df.columns)}")
print(f"   Graphs     : {df[GROUP].nunique():,}")

# تعداد subgroup در هر graph
group_counts = df.groupby(GROUP).size()

print(f"   Min rows/graph : {group_counts.min()}")
print(f"   Max rows/graph : {group_counts.max()}")
print(f"   Mean rows/graph: {group_counts.mean():.2f}")


# ======================================================================
# بررسی ساختار دیتاست
# ======================================================================

print("\n" + "=" * 78)
print("DATASET STRUCTURE CHECK")
print("=" * 78)

print("\n📌 Topology features:")
for col in TOPOLOGY_FEATURES:
    print(f"   ✓ {col}")

print("\n🚫 Explicitly excluded features:")
excluded_features = [
    "distance",
    "subgroup_id",
    "N2",
    "N3",
    "N4",
    "N5",
    "A1",
    "N2_X",
    "N2_Y",
    "N2_Z",
    "single_unique",
    "single_unique_count",
    "total_weight2",
    "stabilizer_size",
    "stabilizer_generators",
    "valid",
    "reason",
    "code",
    "graph_edges",
]

for col in excluded_features:
    if col in df.columns:
        print(f"   ✗ {col}")


# ======================================================================
# حذف NaN
# ======================================================================

required_cols = TOPOLOGY_FEATURES + [TARGET, GROUP]

df_ml = df.dropna(subset=required_cols).copy()

print("\n" + "=" * 78)
print("FINAL DATASET")
print("=" * 78)

print(f"   Original rows : {len(df):,}")
print(f"   Final rows    : {len(df_ml):,}")
print(f"   Graphs        : {df_ml[GROUP].nunique():,}")


# ======================================================================
# هدف
# ======================================================================

y = df_ml[TARGET].to_numpy(dtype=float)

total_variance = np.var(y, ddof=0)

print(f"\n📊 Target statistics:")
print(f"   Mean P_L     : {np.mean(y):.12e}")
print(f"   Std P_L      : {np.std(y):.12e}")
print(f"   Variance P_L : {total_variance:.12e}")
print(f"   Min P_L      : {np.min(y):.12e}")
print(f"   Max P_L      : {np.max(y):.12e}")


# ======================================================================
# X و Groups
# ======================================================================

X = df_ml[TOPOLOGY_FEATURES].to_numpy(dtype=float)

groups = df_ml[GROUP].to_numpy()


# ======================================================================
# بررسی ویژگی‌های topology
# ======================================================================

print("\n" + "=" * 78)
print("TOPOLOGY FEATURE CHECK")
print("=" * 78)

print("\nUnique values:")

for col in TOPOLOGY_FEATURES:
    n_unique = df_ml[col].nunique()
    print(f"   {col:20s}: {n_unique:4d} unique values")


# ======================================================================
# Graph-level variance decomposition
# ======================================================================

print("\n" + "=" * 78)
print("WITHIN-GRAPH VARIANCE ANALYSIS")
print("=" * 78)

# میانگین P_L برای هر graph
graph_stats = (
    df_ml
    .groupby(GROUP)[TARGET]
    .agg(
        graph_mean="mean",
        graph_variance="var",
        graph_std="std",
        n_subgroups="count"
    )
)

# برای گراف‌هایی با فقط یک observation:
# variance = NaN
graph_stats["graph_variance"] = graph_stats["graph_variance"].fillna(0.0)
graph_stats["graph_std"] = graph_stats["graph_std"].fillna(0.0)

# ----------------------------------------------------------------------
# Between-graph variance
# ----------------------------------------------------------------------

graph_means = graph_stats["graph_mean"].to_numpy()

between_graph_variance = np.var(graph_means, ddof=0)

# ----------------------------------------------------------------------
# Within-graph variance
#
# E[Var(Y|G)]
# ----------------------------------------------------------------------

within_graph_variance = (
    graph_stats["graph_variance"]
    .to_numpy()
    .mean()
)

# ----------------------------------------------------------------------
# Law of total variance
#
# Var(Y) = Var(E[Y|G]) + E[Var(Y|G)]
#
# نکته:
# در حالت دقیق با گروه‌های دارای اندازه مساوی، این decomposition
# مستقیماً با population variance کل منطبق می‌شود.
# دیتاست حاضر 31 subgroup برای هر graph دارد.
# ----------------------------------------------------------------------

variance_sum = (
    between_graph_variance +
    within_graph_variance
)

print(f"\nTotal variance:")
print(f"   Var(P_L)                  = {total_variance:.12e}")

print(f"\nBetween-graph variance:")
print(f"   Var(E[P_L | G])          = {between_graph_variance:.12e}")

print(f"\nWithin-graph variance:")
print(f"   E[Var(P_L | G)]          = {within_graph_variance:.12e}")

print(f"\nVariance sum:")
print(f"   Between + Within         = {variance_sum:.12e}")

print(f"\nDifference:")
print(f"   Total - decomposition    = "
      f"{total_variance - variance_sum:.12e}")


# ======================================================================
# درصد واریانس
# ======================================================================

between_fraction = (
    between_graph_variance / total_variance
    if total_variance > 0
    else np.nan
)

within_fraction = (
    within_graph_variance / total_variance
    if total_variance > 0
    else np.nan
)

print("\n📈 Variance fractions:")

print(f"   Between-graph fraction : "
      f"{between_fraction:.6f} "
      f"({100 * between_fraction:.2f}%)")

print(f"   Within-graph fraction  : "
      f"{within_fraction:.6f} "
      f"({100 * within_fraction:.2f}%)")


# ======================================================================
# Graph-only theoretical R² ceiling
# ======================================================================

print("\n" + "=" * 78)
print("GRAPH-LEVEL R² CEILING")
print("=" * 78)

# اگر یک مدل فقط graph-level information داشته باشد،
# نمی‌تواند variation بین subgroupهای یک graph را بازسازی کند.
#
# بهترین deterministic predictor در سطح graph:
#
#     f(G) = E[P_L | G]
#
# بنابراین explained variance آن:
#
#     Var(E[P_L|G])
#
# و upper bound تقریبی:
#
#     R²_max(graph identity)
#       = Var(E[P_L|G]) / Var(P_L)

r2_graph_ceiling = between_graph_variance / total_variance

print(f"""
   Theoretical graph-level ceiling:

   R²_max ≈ Var(E[P_L | G]) / Var(P_L)

   R²_max = {r2_graph_ceiling:.6f}

   This means that a predictor based ONLY on graph-level information
   cannot explain the irreducible within-graph subgroup variation.
""")


# ======================================================================
# مدل‌ها
# ======================================================================

models = {

    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        random_state=RANDOM_STATE
    )
}


# ======================================================================
# Graph-disjoint 5-fold CV
# ======================================================================

print("\n" + "=" * 78)
print("GRAPH-DISJOINT 5-FOLD CROSS-VALIDATION")
print("=" * 78)

print("""
⚠ IMPORTANT:
Each graph_id is assigned entirely to either TRAIN or TEST.

Therefore:
    no graph contributes rows to both sets.
    
This prevents subgroup-level leakage through repeated graph topology.
""")


gkf = GroupKFold(
    n_splits=N_SPLITS
)


all_results = {}


for model_name, base_model in models.items():

    print("\n" + "-" * 78)
    print(f"🧠 MODEL: {model_name}")
    print("-" * 78)

    fold_results = []

    start_total = time.time()

    for fold, (train_idx, test_idx) in enumerate(
        gkf.split(X, y, groups=groups),
        start=1
    ):

        X_train = X[train_idx]
        X_test = X[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        groups_train = groups[train_idx]
        groups_test = groups[test_idx]

        # --------------------------------------------------------------
        # leakage check
        # --------------------------------------------------------------

        train_graphs = set(groups_train)
        test_graphs = set(groups_test)

        overlap = train_graphs.intersection(test_graphs)

        if len(overlap) != 0:
            raise RuntimeError(
                f"DATA LEAKAGE DETECTED in fold {fold}: "
                f"{len(overlap)} graphs overlap."
            )

        # --------------------------------------------------------------
        # train
        # --------------------------------------------------------------

        model = base_model

        start = time.time()

        model.fit(X_train, y_train)

        elapsed = time.time() - start

        # --------------------------------------------------------------
        # prediction
        # --------------------------------------------------------------

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # --------------------------------------------------------------
        # metrics
        # --------------------------------------------------------------

        r2_train = r2_score(
            y_train,
            y_pred_train
        )

        r2_test = r2_score(
            y_test,
            y_pred_test
        )

        mse_test = mean_squared_error(
            y_test,
            y_pred_test
        )

        rmse_test = np.sqrt(
            mse_test
        )

        mae_test = mean_absolute_error(
            y_test,
            y_pred_test
        )

        fold_results.append({
            "fold": fold,
            "train_rows": len(train_idx),
            "test_rows": len(test_idx),
            "train_graphs": len(train_graphs),
            "test_graphs": len(test_graphs),
            "R2_train": r2_train,
            "R2_test": r2_test,
            "MSE_test": mse_test,
            "RMSE_test": rmse_test,
            "MAE_test": mae_test,
            "time_sec": elapsed
        })

        print(
            f"\n   Fold {fold}:"
            f"\n      Train rows  : {len(train_idx):,}"
            f"\n      Test rows   : {len(test_idx):,}"
            f"\n      Train graphs: {len(train_graphs):,}"
            f"\n      Test graphs : {len(test_graphs):,}"
            f"\n      R² Train    : {r2_train:.6f}"
            f"\n      R² Test     : {r2_test:.6f}"
            f"\n      RMSE Test   : {rmse_test:.6e}"
            f"\n      MAE Test    : {mae_test:.6e}"
            f"\n      Time        : {elapsed:.2f}s"
        )

    # ==================================================================
    # خلاصه مدل
    # ==================================================================

    results_df = pd.DataFrame(
        fold_results
    )

    all_results[model_name] = results_df

    total_time = time.time() - start_total

    mean_r2 = results_df["R2_test"].mean()
    std_r2 = results_df["R2_test"].std(ddof=1)

    mean_rmse = results_df["RMSE_test"].mean()
    mean_mae = results_df["MAE_test"].mean()

    print("\n   📊 MODEL SUMMARY")

    print(
        f"      Mean Graph-Disjoint R² : "
        f"{mean_r2:.6f} ± {std_r2:.6f}"
    )

    print(
        f"      Mean RMSE              : "
        f"{mean_rmse:.6e}"
    )

    print(
        f"      Mean MAE               : "
        f"{mean_mae:.6e}"
    )

    print(
        f"      Total time             : "
        f"{total_time:.2f}s"
    )


# ======================================================================
# مقایسه مدل‌ها
# ======================================================================

print("\n" + "=" * 78)
print("FINAL MODEL COMPARISON")
print("=" * 78)

summary_rows = []

for model_name, results_df in all_results.items():

    summary_rows.append({
        "Model": model_name,
        "Mean_R2": results_df["R2_test"].mean(),
        "Std_R2": results_df["R2_test"].std(ddof=1),
        "Mean_RMSE": results_df["RMSE_test"].mean(),
        "Mean_MAE": results_df["MAE_test"].mean()
    })

summary_df = pd.DataFrame(summary_rows)

print("\n")

for _, row in summary_df.iterrows():

    print(
        f"{row['Model']:22s} : "
        f"R² = {row['Mean_R2']:.6f} "
        f"± {row['Std_R2']:.6f} | "
        f"RMSE = {row['Mean_RMSE']:.6e} | "
        f"MAE = {row['Mean_MAE']:.6e}"
    )


# ======================================================================
# Feature importance
# ======================================================================

print("\n" + "=" * 78)
print("FEATURE IMPORTANCE")
print("=" * 78)

# برای اهمیت ویژگی، یک RF روی کل داده صرفاً به منظور
# descriptive analysis آموزش داده می‌شود.
#
# توجه:
# این مدل برای performance evaluation استفاده نمی‌شود.
# Performance همان Graph-disjoint CV است.

rf_full = RandomForestRegressor(
    n_estimators=500,
    max_depth=15,
    min_samples_split=5,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

rf_full.fit(X, y)

importance = rf_full.feature_importances_

importance_df = pd.DataFrame({
    "feature": TOPOLOGY_FEATURES,
    "importance": importance
}).sort_values(
    "importance",
    ascending=False
)

for _, row in importance_df.iterrows():

    print(
        f"   {row['feature']:20s}: "
        f"{row['importance']:.6f}"
    )


# ======================================================================
# ذخیره نتایج
# ======================================================================

print("\n" + "=" * 78)
print("SAVING RESULTS")
print("=" * 78)


# ----------------------------------------------------------------------
# variance decomposition
# ----------------------------------------------------------------------

variance_report = pd.DataFrame([{
    "total_variance": total_variance,
    "between_graph_variance": between_graph_variance,
    "within_graph_variance": within_graph_variance,
    "between_fraction": between_fraction,
    "within_fraction": within_fraction,
    "graph_R2_ceiling": r2_graph_ceiling,
    "n_graphs": df_ml[GROUP].nunique(),
    "n_rows": len(df_ml)
}])

variance_report.to_csv(
    "topology_variance_decomposition.csv",
    index=False
)


# ----------------------------------------------------------------------
# graph statistics
# ----------------------------------------------------------------------

graph_stats.to_csv(
    "within_graph_statistics.csv"
)


# ----------------------------------------------------------------------
# feature importance
# ----------------------------------------------------------------------

importance_df.to_csv(
    "topology_feature_importance.csv",
    index=False
)


# ----------------------------------------------------------------------
# model summary
# ----------------------------------------------------------------------

summary_df.to_csv(
    "graph_disjoint_model_summary.csv",
    index=False
)


# ----------------------------------------------------------------------
# fold results
# ----------------------------------------------------------------------

for model_name, results_df in all_results.items():

    safe_name = (
        model_name
        .lower()
        .replace(" ", "_")
    )

    results_df.to_csv(
        f"graph_disjoint_{safe_name}_folds.csv",
        index=False
    )


# ======================================================================
# FINAL REPORT
# ======================================================================

best_row = summary_df.loc[
    summary_df["Mean_R2"].idxmax()
]

print("\n" + "=" * 78)
print("FINAL REPORT")
print("=" * 78)

print(f"""
📊 DATASET
   Rows                  : {len(df_ml):,}
   Graphs                : {df_ml[GROUP].nunique():,}
   Target                : P_L

🧩 TOPOLOGY FEATURES
   Number of features    : {len(TOPOLOGY_FEATURES)}
   Features:
      {", ".join(TOPOLOGY_FEATURES)}

🔒 LEAKAGE CONTROL
   Split type             : Graph-disjoint
   Number of folds        : {N_SPLITS}
   Shared graphs          : 0

📈 VARIANCE DECOMPOSITION
   Total variance         : {total_variance:.12e}
   Between-graph variance : {between_graph_variance:.12e}
   Within-graph variance  : {within_graph_variance:.12e}

   Between-graph fraction : {between_fraction:.6f}
   Within-graph fraction  : {within_fraction:.6f}

📐 GRAPH-LEVEL R² CEILING
   R² ceiling              : {r2_graph_ceiling:.6f}

🏆 BEST MODEL
   Model                   : {best_row["Model"]}
   Graph-disjoint R²       : {best_row["Mean_R2"]:.6f}
   R² std                  : {best_row["Std_R2"]:.6f}
   Mean RMSE               : {best_row["Mean_RMSE"]:.6e}
   Mean MAE                : {best_row["Mean_MAE"]:.6e}

📁 OUTPUT FILES
   topology_variance_decomposition.csv
   within_graph_statistics.csv
   topology_feature_importance.csv
   graph_disjoint_model_summary.csv
   graph_disjoint_random_forest_folds.csv
   graph_disjoint_gradient_boosting_folds.csv
""")

print("=" * 78)
print("✅ ANALYSIS COMPLETE")
print("=" * 78)