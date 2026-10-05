"""
اسکریپت ساخت دیتاست کامل ML برای QEC Discovery
جایگزین ml_stage_3_4.py
"""

import csv
import itertools
import time
from functools import lru_cache
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# ============================================================
# 1. توابع پائولی (از step_56)
# ============================================================

PAULI_MUL_TABLE = {
    ("I", "I"): "I", ("I", "X"): "X", ("I", "Y"): "Y", ("I", "Z"): "Z",
    ("X", "I"): "X", ("Y", "I"): "Y", ("Z", "I"): "Z",
    ("X", "X"): "I", ("Y", "Y"): "I", ("Z", "Z"): "I",
    ("X", "Y"): "Z", ("Y", "X"): "Z",
    ("Y", "Z"): "X", ("Z", "Y"): "X",
    ("Z", "X"): "Y", ("X", "Z"): "Y",
}

def pauli_mul(p1, p2):
    return "".join(PAULI_MUL_TABLE[(a, b)] for a, b in zip(p1, p2))

def pauli_to_symplectic(pauli):
    n = len(pauli)
    z = np.zeros(n, dtype=np.uint8)
    x = np.zeros(n, dtype=np.uint8)
    for i, p in enumerate(pauli):
        if p == 'X':
            x[i] = 1
        elif p == 'Z':
            z[i] = 1
        elif p == 'Y':
            x[i] = 1
            z[i] = 1
    return np.concatenate([z, x])

def symplectic_product(p1, p2):
    v1 = pauli_to_symplectic(p1)
    v2 = pauli_to_symplectic(p2)
    n = len(p1)
    z1, x1 = v1[:n], v1[n:]
    z2, x2 = v2[:n], v2[n:]
    return int((np.dot(z1, x2) + np.dot(x1, z2)) % 2)

def commute(p1, p2):
    return symplectic_product(p1, p2) == 0

# ============================================================
# 2. توابع گروه پایدارساز (با کش)
# ============================================================

@lru_cache(maxsize=16384)
def build_stabilizer_group_cached(generators_tuple):
    """ساخت گروه پایدارساز با کش"""
    r = len(generators_tuple)
    group = set()
    for mask in range(1 << r):
        current = "I" * len(generators_tuple[0])
        for i, gen in enumerate(generators_tuple):
            if mask & (1 << i):
                current = pauli_mul(current, gen)
        group.add(current)
    return frozenset(group)

@lru_cache(maxsize=16384)
def syndrome_cached(error, stabilizers_tuple):
    """محاسبه‌ی سندرم با کش"""
    return tuple(symplectic_product(error, stab) for stab in stabilizers_tuple)

@lru_cache(maxsize=16384)
def logical_cosets_cached(stabilizers_tuple):
    """یافتن کاست‌های منطقی با کش"""
    n = len(stabilizers_tuple[0])
    group = build_stabilizer_group_cached(stabilizers_tuple)
    all_paulis = [''.join(p) for p in itertools.product(['I','X','Y','Z'], repeat=n)]
    normalizer = [e for e in all_paulis if all(commute(e, s) for s in stabilizers_tuple)]
    non_stab = [e for e in normalizer if e not in group and e != 'I'*n]
    if not non_stab:
        return None
    # انتخاب کم‌وزن‌ترین‌ها
    non_stab.sort(key=lambda p: sum(c!='I' for c in p))
    logical_x = non_stab[0]
    x_coset = frozenset(pauli_mul(logical_x, s) for s in group)
    remaining = [e for e in non_stab if e not in x_coset]
    if not remaining:
        return None
    logical_z = remaining[0]
    z_coset = frozenset(pauli_mul(logical_z, s) for s in group)
    remaining = [e for e in remaining if e not in z_coset]
    if not remaining:
        return None
    logical_y = remaining[0]
    y_coset = frozenset(pauli_mul(logical_y, s) for s in group)
    return {
        'stabilizer': group,
        'logical_X': x_coset,
        'logical_Z': z_coset,
        'logical_Y': y_coset,
    }

def classify_residual(residual, stabilizers_tuple):
    """طبقه‌بندی residual به stabilizer یا logical"""
    cosets = logical_cosets_cached(stabilizers_tuple)
    if cosets is None:
        return 'unknown'
    if residual in cosets['stabilizer']:
        return 'stabilizer'
    if residual in cosets['logical_X']:
        return 'logical_X'
    if residual in cosets['logical_Z']:
        return 'logical_Z'
    if residual in cosets['logical_Y']:
        return 'logical_Y'
    return 'unknown'

# ============================================================
# 3. محاسبه‌ی failures برای وزن‌های ۲ تا ۵ (با کش)
# ============================================================

@lru_cache(maxsize=8192)
def weight_failures_cached(stabilizers_tuple, w):
    """تعداد failures برای یک وزن مشخص (۲ تا ۵)"""
    n = len(stabilizers_tuple[0])
    stabilizers = list(stabilizers_tuple)
    group = build_stabilizer_group_cached(stabilizers_tuple)
    
    # دیکشنری decoder (کم‌وزن‌ترین تصحیح برای هر سندرم)
    decoder = {}
    for error in [''.join(p) for p in itertools.product(['I','X','Y','Z'], repeat=n)]:
        s = syndrome_cached(error, stabilizers_tuple)
        weight = sum(c != 'I' for c in error)
        if s not in decoder or weight < decoder[s][1]:
            decoder[s] = (error, weight)
    decoder = {s: item[0] for s, item in decoder.items()}
    
    failures = 0
    # تولید خطاهای با وزن دقیقاً w
    for positions in itertools.combinations(range(n), w):
        for paulis in itertools.product(['X','Y','Z'], repeat=w):
            error = list('I' * n)
            for pos, p in zip(positions, paulis):
                error[pos] = p
            error = ''.join(error)
            s = syndrome_cached(error, stabilizers_tuple)
            correction = decoder[s]
            residual = pauli_mul(error, correction)
            result = classify_residual(residual, stabilizers_tuple)
            if result != 'stabilizer':
                failures += 1
    return failures

# ============================================================
# 4. پردازش یک ردیف از دیتاست
# ============================================================

def process_row(row):
    """دریافت stabilizer generators و محاسبه‌ی N2 تا N5"""
    gen_str = row['stabilizer_generators']
    stabilizers = tuple(gen_str.split(';'))
    
    # محاسبه با کش
    N2 = weight_failures_cached(stabilizers, 2)
    N3 = weight_failures_cached(stabilizers, 3)
    N4 = weight_failures_cached(stabilizers, 4)
    N5 = weight_failures_cached(stabilizers, 5)
    
    return {
        'graph_id': row['graph_id'],
        'subgroup_id': row['subgroup_id'],
        'distance': row['distance'],
        'N2': N2,
        'N3': N3,
        'N4': N4,
        'N5': N5,
    }

# ============================================================
# 5. محاسبه‌ی P_L(p)
# ============================================================

def compute_PL(N2, N3, N4, N5, p=0.01):
    n = 5
    PL = 0.0
    for w, Nw in [(2, N2), (3, N3), (4, N4), (5, N5)]:
        if Nw is not None and Nw > 0:
            PL += Nw * (p/3)**w * (1-p)**(n-w)
    return PL

# ============================================================
# 6. MAIN
# ============================================================

def main():
    print("=" * 70)
    print("ساخت دیتاست کامل ML برای QEC Discovery")
    print("=" * 70)
    
    # بارگذاری دیتاست موجود
    input_file = "stage_58_optimized_landscape.csv"
    print(f"\n📂 بارگذاری دیتاست از {input_file} ...")
    df = pd.read_csv(input_file)
    print(f"   تعداد ردیف‌ها: {len(df)}")
    
    # فیلتر کردن ردیف‌های معتبر (با stabilizer_generators)
    df = df.dropna(subset=['stabilizer_generators'])
    print(f"   پس از حذف NaN در stabilizer_generators: {len(df)} ردیف")
    
    # محدود کردن به کدهای معتبر با distance مشخص
    df = df[df['distance'].notna()]
    print(f"   کدهای با distance مشخص: {len(df)} ردیف")
    
    # ============================================================
    # محاسبه‌ی N2 تا N5 برای هر ردیف (با چندپردازشی)
    # ============================================================
    print("\n🔄 محاسبه‌ی N2 تا N5 برای هر کد...")
    print("   (این کار ممکن است چند دقیقه طول بکشد)")
    
    rows = df.to_dict('records')
    
    # استفاده از multiprocessing برای سرعت
    num_cores = min(cpu_count(), 8)
    print(f"   استفاده از {num_cores} هسته")
    
    start_time = time.time()
    with Pool(processes=num_cores) as pool:
        results = pool.map(process_row, rows)
    elapsed = time.time() - start_time
    print(f"   ✅ محاسبه در {elapsed:.1f} ثانیه انجام شد")
    
    # تبدیل به DataFrame
    results_df = pd.DataFrame(results)
    
    # اضافه کردن ستون‌های جدید به دیتاست اصلی
    for col in ['N2', 'N3', 'N4', 'N5']:
        df[col] = results_df[col].values
    
    # محاسبه‌ی P_L برای p=0.01
    p = 0.01
    df['P_L'] = df.apply(lambda r: compute_PL(r['N2'], r['N3'], r['N4'], r['N5'], p), axis=1)
    
    # ============================================================
    # آماده‌سازی ویژگی‌ها برای ML
    # ============================================================
    print("\n📊 آماده‌سازی ویژگی‌ها برای مدل ML...")
    
    # ستون‌های عددی که به‌عنوان ویژگی استفاده می‌شوند
    feature_cols = [
        'edges', 'min_degree', 'max_degree', 'mean_degree',
        'degree_variance', 'diameter', 'triangles', 'cycle_rank',
        'subgroup_id', 'distance'
    ]
    
    # حذف ردیف‌هایی که ویژگی‌هایشان NaN است
    df_ml = df.dropna(subset=feature_cols + ['P_L'])
    print(f"   ردیف‌های نهایی برای ML: {len(df_ml)}")
    
    X = df_ml[feature_cols].values
    y = df_ml['P_L'].values
    
    # بررسی واریانس هدف
    y_var = np.var(y)
    print(f"   واریانس هدف: {y_var:.6f}")
    if y_var < 1e-12:
        print("   ⚠️ هشدار: واریانس هدف تقریباً صفر است! R² تعریف‌نشده خواهد بود.")
        print("   احتمالاً همه‌ی کدها P_L مشابهی دارند.")
        # ادامه می‌دهیم ولی انتظار R² معنی‌دار نداریم
    
    # تقسیم داده‌ها
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # ============================================================
    # آموزش مدل
    # ============================================================
    print("\n🧠 آموزش مدل Random Forest...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # پیش‌بینی
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # معیارها
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    mse_test = mean_squared_error(y_test, y_pred_test)
    
    print(f"\n📈 نتایج:")
    print(f"   R² (Train): {r2_train:.6f}")
    print(f"   R² (Test) : {r2_test:.6f}")
    print(f"   MSE (Test): {mse_test:.6e}")
    
    # ============================================================
    # ذخیره‌ی دیتاست کامل
    # ============================================================
    output_file = "ml_full_dataset.csv"
    df.to_csv(output_file, index=False)
    print(f"\n💾 دیتاست کامل با تمام ستون‌ها در {output_file} ذخیره شد.")
    
    # ============================================================
    # گزارش نهایی
    # ============================================================
    print("\n" + "=" * 70)
    print("🏆 اسکریپت با موفقیت کامل شد")
    print("=" * 70)
    print(f"""
    خلاصه:
    - تعداد کل کدهای معتبر: {len(df)}
    - تعداد کدهای استفاده‌شده در ML: {len(df_ml)}
    - واریانس هدف (P_L): {y_var:.6f}
    - R² روی تست: {r2_test:.6f}
    
    اگر R² نزدیک به ۱ باشد، نشان‌دهنده‌ی قدرت پیش‌بینی مدل است.
    اگر R² نزدیک به صفر باشد، یعنی ویژگی‌ها قدرت تفکیک ندارند.
    """)
    
    # تحلیل توزیع P_L
    print("\n📊 توزیع P_L:")
    print(df['P_L'].describe())
    
    return df, model

if __name__ == "__main__":
    main()