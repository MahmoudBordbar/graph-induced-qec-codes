"""
مرحله ۲۸: Exact Full Polynomial + Weight Enumerator
فایل: step_28_exact_full_polynomial.py
"""

import numpy as np
from itertools import product

# Stabilizerهای [[5,1,3]]
STABILIZERS = [
    "XZZXI",
    "IXZZX",
    "XIXZZ",
    "ZXIXZ"
]

LOGICAL_X = "IIXYX"
LOGICAL_Z = "IIZXZ"

def pauli_to_symplectic(pauli_str):
    n = len(pauli_str)
    z = np.zeros(n, dtype=int)
    x = np.zeros(n, dtype=int)
    for i, p in enumerate(pauli_str):
        if p == 'X':
            x[i] = 1
        elif p == 'Z':
            z[i] = 1
        elif p == 'Y':
            x[i] = 1
            z[i] = 1
    return np.concatenate([z, x])

def symplectic_inner_product(v1, v2):
    n = len(v1) // 2
    z1, x1 = v1[:n], v1[n:]
    z2, x2 = v2[:n], v2[n:]
    return (np.dot(z1, x2) + np.dot(x1, z2)) % 2

def pauli_mul(p1, p2):
    n = len(p1)
    result = []
    phase = 1
    for a, b in zip(p1, p2):
        if a == 'I':
            result.append(b)
        elif b == 'I':
            result.append(a)
        elif a == b:
            result.append('I')
        elif a == 'X' and b == 'Z':
            result.append('Y')
            phase *= -1j
        elif a == 'Z' and b == 'X':
            result.append('Y')
            phase *= 1j
        elif a == 'X' and b == 'Y':
            result.append('Z')
            phase *= 1j
        elif a == 'Y' and b == 'X':
            result.append('Z')
            phase *= -1j
        elif a == 'Y' and b == 'Z':
            result.append('X')
            phase *= -1j
        elif a == 'Z' and b == 'Y':
            result.append('X')
            phase *= 1j
    return ''.join(result), phase

def build_stabilizer_group(generators):
    n = len(generators)
    group = set()
    for mask in range(1 << n):
        current = 'I' * len(generators[0])
        phase = 1
        for i in range(n):
            if mask & (1 << i):
                current, phase = pauli_mul(current, generators[i])
        group.add(current)
    return group

def compute_syndrome(error, stabilizers):
    syndrome = []
    v_error = pauli_to_symplectic(error)
    for stab in stabilizers:
        v_stab = pauli_to_symplectic(stab)
        s = symplectic_inner_product(v_error, v_stab)
        syndrome.append(s)
    return tuple(syndrome)

def build_minimum_weight_decoder(stabilizers):
    all_paulis = [''.join(p) for p in product(['I', 'X', 'Y', 'Z'], repeat=5)]
    decoder = {}
    for error in all_paulis:
        if error == 'I' * 5:
            continue
        syndrome = compute_syndrome(error, stabilizers)
        if sum(syndrome) == 0:
            continue
        weight = sum(1 for c in error if c != 'I')
        if syndrome not in decoder or weight < decoder[syndrome]['weight']:
            decoder[syndrome] = {'error': error, 'weight': weight}
    return decoder

def check_logical_coset(error, logical_x, logical_z, stabilizers):
    stab_group = build_stabilizer_group(stabilizers)
    if error in stab_group:
        return 'stabilizer'
    x_coset, _ = pauli_mul(error, logical_x)
    if x_coset in stab_group:
        return 'logical_X'
    z_coset, _ = pauli_mul(error, logical_z)
    if z_coset in stab_group:
        return 'logical_Z'
    logical_y, _ = pauli_mul(logical_x, logical_z)
    y_coset, _ = pauli_mul(error, logical_y)
    if y_coset in stab_group:
        return 'logical_Y'
    return 'unknown'

def compute_weight_enumerator():
    """محاسبه تعداد failures برای هر وزن"""
    all_paulis = [''.join(p) for p in product(['I', 'X', 'Y', 'Z'], repeat=5)]
    decoder = build_minimum_weight_decoder(STABILIZERS)
    
    failures_by_weight = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    total_by_weight = {0: 1, 1: 15, 2: 90, 3: 270, 4: 405, 5: 243}
    
    for error in all_paulis:
        weight = sum(1 for c in error if c != 'I')
        syndrome = compute_syndrome(error, STABILIZERS)
        
        if sum(syndrome) == 0:
            result = check_logical_coset(error, LOGICAL_X, LOGICAL_Z, STABILIZERS)
            if result != 'stabilizer':
                failures_by_weight[weight] += 1
            continue
        
        if syndrome in decoder:
            correction = decoder[syndrome]['error']
            residual, _ = pauli_mul(error, correction)
            result = check_logical_coset(residual, LOGICAL_X, LOGICAL_Z, STABILIZERS)
            if result != 'stabilizer':
                failures_by_weight[weight] += 1
    
    return failures_by_weight, total_by_weight

def compute_pl_polynomial(failures_by_weight, total_by_weight):
    """محاسبه چندجمله‌ای کامل P_L(p)"""
    print("\n📊 Weight enumerator:")
    print("   weight | failures | total | success |")
    print("   -------|----------|-------|---------|")
    
    for w in range(6):
        failures = failures_by_weight[w]
        total = total_by_weight[w]
        success = total - failures
        print(f"      {w}   |   {failures:4d}   |  {total:4d} |   {success:4d}   |")
    
    # ساخت چندجمله‌ای
    def pl_poly(p):
        result = 0
        for w in range(6):
            failures = failures_by_weight[w]
            if failures > 0:
                result += failures * (1 - p)**(5 - w) * (p/3)**w
        return result
    
    return pl_poly

def run_analysis():
    """اجرای تحلیل کامل"""
    print("="*70)
    print("مرحله ۲۸: Exact Full Polynomial + Weight Enumerator")
    print("="*70)
    
    failures_by_weight, total_by_weight = compute_weight_enumerator()
    pl_poly = compute_pl_polynomial(failures_by_weight, total_by_weight)
    
    print("\n📊 چندجمله‌ای P_L(p):")
    print(f"   P_L(p) = ", end="")
    terms = []
    for w in range(6):
        f = failures_by_weight[w]
        if f > 0:
            terms.append(f"{f} (1-p)^({5-w}) (p/3)^{w}")
    print(" + ".join(terms))
    
    # محاسبه برای چند نقطه
    print("\n📊 مقادیر عددی:")
    for p in [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]:
        pl = pl_poly(p)
        print(f"   p = {p:.3f} → P_L = {pl:.6f}")
    
    print("\n" + "="*70)
    print("📌 نتیجه نهایی:")
    print("   ✅ Exact full polynomial P_L(p) محاسبه شد.")
    print("   ✅ آماده برای Stim validation.")
    print("="*70)

if __name__ == "__main__":
    run_analysis()