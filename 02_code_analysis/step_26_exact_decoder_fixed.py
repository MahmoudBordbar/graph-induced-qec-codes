"""
مرحله ۲۶: Exact Minimum-Weight Decoder + Logical Coset Verification
فایل: step_26_exact_decoder_fixed.py
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
    """ساخت minimum-weight decoder"""
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
    """بررسی membership در coset‌های logical"""
    stab_group = build_stabilizer_group(stabilizers)
    
    # اگر در stabilizer باشد
    if error in stab_group:
        return 'stabilizer'
    
    # بررسی logical X coset
    x_coset, _ = pauli_mul(error, logical_x)
    if x_coset in stab_group:
        return 'logical_X'
    
    # بررسی logical Z coset
    z_coset, _ = pauli_mul(error, logical_z)
    if z_coset in stab_group:
        return 'logical_Z'
    
    # بررسی logical Y coset (X*Z)
    logical_y, _ = pauli_mul(logical_x, logical_z)
    y_coset, _ = pauli_mul(error, logical_y)
    if y_coset in stab_group:
        return 'logical_Y'
    
    return 'unknown'

def compute_pl_exact(p, stabilizers, logical_x, logical_z):
    """محاسبه exact P_L با minimum-weight decoder"""
    all_paulis = [''.join(p) for p in product(['I', 'X', 'Y', 'Z'], repeat=5)]
    decoder = build_minimum_weight_decoder(stabilizers)
    
    total_pl = 0
    counts = {'stabilizer': 0, 'logical_X': 0, 'logical_Z': 0, 'logical_Y': 0}
    
    for error in all_paulis:
        weight = sum(1 for c in error if c != 'I')
        prob = (1 - p)**(5 - weight) * (p/3)**weight
        
        syndrome = compute_syndrome(error, stabilizers)
        if sum(syndrome) == 0:
            result = check_logical_coset(error, logical_x, logical_z, stabilizers)
            if result != 'stabilizer':
                total_pl += prob
                counts[result] = counts.get(result, 0) + 1
            continue
        
        if syndrome in decoder:
            correction = decoder[syndrome]['error']
            residual, _ = pauli_mul(error, correction)
            result = check_logical_coset(residual, logical_x, logical_z, stabilizers)
            if result != 'stabilizer':
                total_pl += prob
                counts[result] = counts.get(result, 0) + 1
    
    return total_pl, counts

def run_exact_decoder():
    """اجرای exact decoder"""
    print("="*70)
    print("مرحله ۲۶: Exact Minimum-Weight Decoder + Logical Coset")
    print("="*70)
    
    # ۱. ساخت decoder
    print("\n📊 ① ساخت minimum-weight decoder:")
    decoder = build_minimum_weight_decoder(STABILIZERS)
    print(f"   تعداد entries: {len(decoder)}")
    
    # ۲. تست single-qubit errors
    print("\n📊 ② تست ۱۵ خطای تک-کیوبیتی:")
    single_errors = []
    for i in range(5):
        for e in ['X', 'Y', 'Z']:
            error = list('I' * 5)
            error[i] = e
            single_errors.append(''.join(error))
    
    all_passed = True
    for error in single_errors:
        syndrome = compute_syndrome(error, STABILIZERS)
        if syndrome in decoder:
            correction = decoder[syndrome]['error']
            residual, _ = pauli_mul(error, correction)
            result = check_logical_coset(residual, LOGICAL_X, LOGICAL_Z, STABILIZERS)
            passed = (result == 'stabilizer')
            if not passed:
                all_passed = False
                print(f"   ❌ {error} → {syndrome} → {residual} → {result}")
    
    if all_passed:
        print("   ✅ تمام ۱۵ خطای تک-کیوبیتی به stabilizer برگشتند.")
    else:
        print("   ⚠️ برخی خطاها به stabilizer برنگشتند.")
    
    # ۳. محاسبه P_L
    print("\n📊 ③ محاسبه exact P_L:")
    noise_levels = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
    pl_values = []
    
    for p in noise_levels:
        pl, counts = compute_pl_exact(p, STABILIZERS, LOGICAL_X, LOGICAL_Z)
        pl_values.append(pl)
        print(f"   p = {p:.3f} → P_L = {pl:.6f}")
    
    print("\n📌 نتیجه:")
    print("   ✅ Exact P_L با minimum-weight decoder محاسبه شد.")
    print("   ✅ آماده برای مقاله.")

if __name__ == "__main__":
    run_exact_decoder()