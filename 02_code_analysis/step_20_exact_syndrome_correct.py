"""
مرحله ۲۰: محاسبه exact syndrome با symplectic algebra
فایل: step_20_exact_syndrome_correct.py
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

def pauli_to_symplectic(pauli_str):
    """تبدیل رشته Pauli به بردار symplectic (z, x)"""
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

def compute_syndrome(error, stabilizers):
    """
    محاسبه syndrome برای یک خطا با symplectic algebra
    """
    syndrome = []
    v_error = pauli_to_symplectic(error)
    
    for stab in stabilizers:
        v_stab = pauli_to_symplectic(stab)
        s = symplectic_inner_product(v_error, v_stab)
        syndrome.append(s)
    
    return syndrome

def create_single_qubit_error(i, error_type):
    """ساخت یک خطای تک-کیوبیتی"""
    n = 5
    pauli_str = ['I'] * n
    pauli_str[i] = error_type
    return ''.join(pauli_str)

def run_exact_syndrome_analysis():
    """اجرای تحلیل exact syndrome"""
    print("="*70)
    print("مرحله ۲۰: Exact Syndrome با Symplectic Algebra")
    print("="*70)
    
    print(f"\n📊 Stabilizerهای [[5,1,3]]:")
    for i, stab in enumerate(STABILIZERS):
        print(f"   S{i} = {stab}")
    
    print("\n📊 محاسبه syndrome برای ۱۵ خطای تک-کیوبیتی:")
    print("-"*70)
    print("   Error → Syndrome")
    print("-"*70)
    
    errors = []
    syndromes = []
    
    for i in range(5):
        for error_type in ['X', 'Y', 'Z']:
            error = create_single_qubit_error(i, error_type)
            syndrome = compute_syndrome(error, STABILIZERS)
            errors.append(error)
            syndromes.append(syndrome)
            
            # نمایش
            syndrome_str = ''.join(str(s) for s in syndrome)
            print(f"   {i}{error_type} → {syndrome}")
    
    # بررسی یکتایی
    unique = set(tuple(s) for s in syndromes)
    print("-"*70)
    print(f"\n   تعداد syndromeهای یکتا: {len(unique)}")
    
    # بررسی non-zero بودن
    nonzero = sum(1 for s in syndromes if any(s))
    print(f"   تعداد syndromeهای غیرصفر: {nonzero}")
    
    # نتیجه نهایی
    print("\n" + "="*70)
    print("📌 نتیجه نهایی:")
    if len(unique) == 15 and nonzero == 15:
        print("   ✅ تمام ۱۵ خطای تک-کیوبیتی syndrome یکتا و غیرصفر دارند.")
        print("   ✅ d=3 تأیید شد.")
        print("   ✅ [[5,1,3]] یک کد perfect است.")
        print("   ✅ آماده برای شبیه‌سازی P_L.")
    else:
        print("   ⚠️ نتایج با کد perfect مطابقت ندارد.")
    print("="*70)

if __name__ == "__main__":
    run_exact_syndrome_analysis()