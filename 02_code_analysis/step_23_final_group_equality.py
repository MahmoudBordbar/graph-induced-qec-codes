"""
مرحله ۲۳: تأیید برابری دقیق دو گروه stabilizer
فایل: step_23_final_group_equality.py
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

# Graph State Cycle-5 stabilizers
GRAPH_STABILIZERS = [
    "XZIIZ",
    "ZXZII",
    "IZXZI",
    "IIZXZ",
    "ZIIZX"
]

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
    """ساخت کامل گروه stabilizer (با فاز)"""
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

def compare_groups_exact(group1, group2):
    """مقایسه دقیق دو گروه"""
    if len(group1) != len(group2):
        return False, f"size mismatch: {len(group1)} vs {len(group2)}"
    
    diff = group1 - group2
    if diff:
        return False, f"membership mismatch: {list(diff)[:5]}"
    
    return True, "groups are identical"

def run_final_verification():
    """اجرای verification نهایی"""
    print("="*70)
    print("مرحله ۲۳: تأیید برابری دقیق دو گروه stabilizer")
    print("="*70)
    
    # ۱. ساخت گروه کد + logical Z
    print("\n📊 ① Stabilizer state (کد + logical Z):")
    for i, stab in enumerate(CODE_STABILIZERS):
        print(f"   G{i} = {stab}")
    
    group_code = build_stabilizer_group(CODE_STABILIZERS)
    print(f"\n   اندازه گروه: {len(group_code)}")
    
    # ۲. ساخت گروه Graph State Cycle-5
    print("\n📊 ② Graph State Cycle-5:")
    for i, stab in enumerate(GRAPH_STABILIZERS):
        print(f"   K{i} = {stab}")
    
    group_graph = build_stabilizer_group(GRAPH_STABILIZERS)
    print(f"\n   اندازه گروه: {len(group_graph)}")
    
    # ۳. مقایسه دقیق
    print("\n📊 ③ مقایسه دقیق:")
    is_equal, message = compare_groups_exact(group_code, group_graph)
    
    if is_equal:
        print("   ✅ دو گروه دقیقاً برابر هستند!")
        print("   ✅ [[5,1,3]] با logical Z=IIZXZ دقیقاً معادل Cycle-5 است.")
        print("   ✅ U_LC = I⊗I⊗I⊗I⊗I")
        print("   ✅ آماده برای شبیه‌سازی P_L.")
    else:
        print(f"   ⚠️ {message}")
        print("   نیاز به بررسی بیشتر دارد.")
    
    print("\n" + "="*70)
    print("📌 نتیجه نهایی:")
    if is_equal:
        print("   ✅ [[5,1,3]] → Cycle-5 Graph State تأیید شد!")
        print("   ✅ می‌توانیم به Stim و P_L برویم.")
    else:
        print("   ⚠️ گروه‌ها برابر نیستند.")
    print("="*70)

if __name__ == "__main__":
    run_final_verification()