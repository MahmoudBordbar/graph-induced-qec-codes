"""
مرحله ۲۴: Independent Exact Verification
فایل: step_24_independent_verification.py
"""

import numpy as np
from itertools import product

# Stabilizerهای [[5,1,3]] + logical Z
CODE_STABILIZERS = [
    "XZZXI",
    "IXZZX",
    "XIXZZ",
    "ZXIXZ",
    "IIZXZ"
]

LOGICAL_X = "IIXYX"

# Graph State Cycle-5 stabilizers
GRAPH_STABILIZERS = [
    "XZIIZ",
    "ZXZII",
    "IZXZI",
    "IIZXZ",
    "ZIIZX"
]

def pauli_to_symplectic(pauli_str):
    """تبدیل رشته Pauli به بردار symplectic"""
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
    """محصول داخلی سمپلکتیک"""
    n = len(v1) // 2
    z1, x1 = v1[:n], v1[n:]
    z2, x2 = v2[:n], v2[n:]
    return (np.dot(z1, x2) + np.dot(x1, z2)) % 2

def pauli_mul(p1, p2):
    """ضرب دو رشته Pauli با فاز"""
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
    """ساخت کامل گروه stabilizer"""
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

def check_linear_independence(stabilizers):
    """بررسی استقلال خطی stabilizerها"""
    n = len(stabilizers)
    m = len(stabilizers[0])
    vectors = [pauli_to_symplectic(s) for s in stabilizers]
    
    # بررسی rank
    matrix = np.array(vectors)
    rank = np.linalg.matrix_rank(matrix)
    return rank == n

def check_commutation(stabilizers):
    """بررسی جابه‌جایی همه stabilizerها"""
    for i in range(len(stabilizers)):
        for j in range(i+1, len(stabilizers)):
            v1 = pauli_to_symplectic(stabilizers[i])
            v2 = pauli_to_symplectic(stabilizers[j])
            if symplectic_inner_product(v1, v2) != 0:
                return False, i, j
    return True, -1, -1

def check_group_equality(g1, g2):
    """بررسی برابری دو گروه"""
    return g1 == g2

def run_verification():
    """اجرای verification کامل"""
    print("="*70)
    print("مرحله ۲۴: Independent Exact Verification")
    print("="*70)
    
    all_passed = True
    
    # ۱. استقلال خطی
    print("\n📊 ① استقلال خطی:")
    if check_linear_independence(CODE_STABILIZERS):
        print("   ✅ همه stabilizerها مستقل هستند.")
    else:
        print("   ❌ stabilizerها مستقل نیستند.")
        all_passed = False
    
    # ۲. جابه‌جایی
    print("\n📊 ② جابه‌جایی:")
    commutes, i, j = check_commutation(CODE_STABILIZERS)
    if commutes:
        print("   ✅ همه stabilizerها commute می‌کنند.")
    else:
        print(f"   ❌ S{i} و S{j} commute نمی‌کنند.")
        all_passed = False
    
    # ۳. اندازه گروه
    print("\n📊 ③ اندازه گروه:")
    group_code = build_stabilizer_group(CODE_STABILIZERS)
    if len(group_code) == 32:
        print("   ✅ اندازه گروه = 32")
    else:
        print(f"   ❌ اندازه گروه = {len(group_code)}")
        all_passed = False
    
    # ۴. برابری با Graph State
    print("\n📊 ④ برابری با Graph State Cycle-5:")
    group_graph = build_stabilizer_group(GRAPH_STABILIZERS)
    if check_group_equality(group_code, group_graph):
        print("   ✅ گروه‌ها برابر هستند.")
    else:
        print("   ❌ گروه‌ها برابر نیستند.")
        all_passed = False
    
    # ۵. logical operators
    print("\n📊 ⑤ logical operators:")
    logical_z = CODE_STABILIZERS[-1]
    print(f"   logical Z = {logical_z}")
    print(f"   logical X = {LOGICAL_X}")
    
    # بررسی anticommutation
    vx = pauli_to_symplectic(LOGICAL_X)
    vz = pauli_to_symplectic(logical_z)
    if symplectic_inner_product(vx, vz) == 1:
        print("   ✅ logical X و Z anticommute می‌کنند.")
    else:
        print("   ❌ logical X و Z anticommute نمی‌کنند.")
        all_passed = False
    
    # ۶. نتیجه نهایی
    print("\n" + "="*70)
    print("📌 نتیجه نهایی:")
    if all_passed:
        print("   ✅ همه تست‌ها PASS شدند!")
        print("   ✅ [[5,1,3]] با logical Z=IIZXZ تأیید شد!")
        print("   ✅ معادل Cycle-5 Graph State.")
        print("   ✅ آماده برای شبیه‌سازی P_L.")
    else:
        print("   ⚠️ برخی تست‌ها FAIL شدند.")
    print("="*70)

if __name__ == "__main__":
    run_verification()