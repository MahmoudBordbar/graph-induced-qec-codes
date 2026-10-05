"""
======================================================================
TEST 01: Complete LC + Permutation Equivalence Check
======================================================================
هدف: تأیید ادعای مقاله که همه 132 کد d=3 با canonical [[5,1,3]]
      تحت LC+Permutation هم‌ارز هستند.

خروجی مورد انتظار:
    Matched: 132
    Unmatched: 0
======================================================================
"""

import numpy as np
import pandas as pd
from itertools import permutations, product
import time
from collections import Counter


# ======================================================================
# Effective LC classes (6 classes = cosets of Pauli in Clifford)
# ======================================================================
EFFECTIVE_LC = [
    {'I': 'I', 'X': 'X', 'Y': 'Y', 'Z': 'Z'},
    {'I': 'I', 'X': 'X', 'Y': 'Z', 'Z': 'Y'},
    {'I': 'I', 'X': 'Z', 'Y': 'Y', 'Z': 'X'},
    {'I': 'I', 'X': 'Y', 'Y': 'X', 'Z': 'Z'},
    {'I': 'I', 'X': 'Y', 'Y': 'Z', 'Z': 'X'},
    {'I': 'I', 'X': 'Z', 'Y': 'X', 'Z': 'Y'},
]


def pauli_mul(p1, p2):
    table = {
        ('I','I'):'I',('I','X'):'X',('I','Y'):'Y',('I','Z'):'Z',
        ('X','I'):'X',('Y','I'):'Y',('Z','I'):'Z',
        ('X','X'):'I',('Y','Y'):'I',('Z','Z'):'I',
        ('X','Y'):'Z',('Y','X'):'Z',
        ('Y','Z'):'X',('Z','Y'):'X',
        ('Z','X'):'Y',('X','Z'):'Y',
    }
    return ''.join(table[(a, b)] for a, b in zip(p1, p2))


def build_group(generators):
    r = len(generators)
    group = set()
    for mask in range(1 << r):
        current = 'I' * len(generators[0])
        for i, gen in enumerate(generators):
            if mask & (1 << i):
                current = pauli_mul(current, gen)
        group.add(current)
    return frozenset(group)


def pauli_weight(p):
    return sum(1 for c in p if c != 'I')


def apply_perm(pauli, perm):
    return ''.join(pauli[perm[i]] for i in range(len(perm)))


def apply_lc(pauli, lc_combo):
    return ''.join(lc_combo[i][pauli[i]] for i in range(len(pauli)))


def main():
    print("=" * 70)
    print("TEST 01: Complete LC + Permutation Equivalence Check")
    print("=" * 70)

    # بارگذاری دیتاست
    try:
        df = pd.read_csv("ml_full_dataset.csv")
    except FileNotFoundError:
        print("❌ ml_full_dataset.csv پیدا نشد!")
        print("   ابتدا build_proper_ml_dataset.py را اجرا کنید.")
        return

    d3 = df[df['distance'] == 3].copy()
    print(f"\n📂 Total codes with d=3: {len(d3)}")
    print(f"   Expected: 132")

    if len(d3) != 132:
        print(f"⚠️  Warning: expected 132, got {len(d3)}")

    # Canonical
    CANONICAL = ["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"]
    canonical_group = build_group(CANONICAL)

    print(f"\n📊 Canonical code:")
    print(f"   Generators: {CANONICAL}")
    print(f"   Group size: {len(canonical_group)}")

    # همه permutationها و LCها را از قبل بسازیم
    all_perms = list(permutations(range(5)))
    all_lc = list(product(EFFECTIVE_LC, repeat=5))
    print(f"\n📊 Search space per code:")
    print(f"   Permutations: {len(all_perms)}")
    print(f"   LC combos:    {len(all_lc)}")
    print(f"   Total:        {len(all_perms) * len(all_lc):,}")

    matched = []
    unmatched = []
    errors = []

    t0 = time.time()

    for idx, (_, row) in enumerate(d3.iterrows(), start=1):
        gen_str = row['stabilizer_generators']
        if pd.isna(gen_str):
            errors.append((row['graph_id'], 'NaN generators'))
            continue

        generators = gen_str.split(';')
        if len(generators) != 4:
            errors.append((row['graph_id'], f'len={len(generators)}'))
            continue

        # چک اول: آیا خودش معادل است؟
        original_group = build_group(generators)
        if original_group == canonical_group:
            matched.append(row['graph_id'])
            continue

        # جستجوی LC+Perm
        found = False
        for perm in all_perms:
            permuted = [apply_perm(g, perm) for g in generators]
            for lc in all_lc:
                transformed = [apply_lc(g, lc) for g in permuted]
                if build_group(transformed) == canonical_group:
                    found = True
                    break
            if found:
                break

        if found:
            matched.append(row['graph_id'])
        else:
            unmatched.append(row['graph_id'])

        # نمایش پیشرفت
        if idx % 10 == 0 or idx == len(d3):
            elapsed = time.time() - t0
            print(f"   Progress: {idx}/{len(d3)} | "
                  f"matched={len(matched)} | "
                  f"unmatched={len(unmatched)} | "
                  f"elapsed={elapsed:.1f}s")

    # ==================================================================
    # گزارش نهایی
    # ==================================================================
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print(f"\n📊 Summary:")
    print(f"   Total codes checked: {len(d3)}")
    print(f"   Matched (equivalent): {len(matched)}")
    print(f"   Unmatched:            {len(unmatched)}")
    print(f"   Errors:               {len(errors)}")

    print(f"\n📌 Article claim:")
    print(f"   All 132 constructions are LC+permutation equivalent.")

    if len(unmatched) == 0 and len(errors) == 0:
        print("\n" + "=" * 70)
        print("🎉 TEST 01 PASSED — Article claim verified!")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("❌ TEST 01 FAILED — Article claim is INCORRECT!")
        print("=" * 70)

        if unmatched:
            print(f"\n⚠️  Unmatched codes ({len(unmatched)}):")
            for gid in unmatched[:20]:
                print(f"   - {gid}")
            if len(unmatched) > 20:
                print(f"   ... and {len(unmatched) - 20} more")

        if errors:
            print(f"\n⚠️  Error codes ({len(errors)}):")
            for gid, err in errors[:20]:
                print(f"   - {gid}: {err}")


if __name__ == "__main__":
    main()