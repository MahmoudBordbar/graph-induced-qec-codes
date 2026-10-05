"""
======================================================================
TEST 03: Temporal Two-Fault Scaling — Verify N₂ = 90R²
======================================================================
هدف: تأیید نتیجه مقاله:
     N₂(D_full; R) = 90R² for R = 1, 2, 3
     و P_L(p, R) = 10R²·p² + O(p³)

روش: Exhaustive enumeration of same-round and cross-round two-fault
     patterns for each R.

خروجی مورد انتظار:
    R  |  Same  |  Cross  |  Total  |  90R²
    1  |    90  |      0  |     90  |    90   ✅
    2  |   180  |    180  |    360  |   360   ✅
    3  |   270  |    540  |    810  |   810   ✅
======================================================================
"""

import numpy as np
from itertools import combinations, product


# ======================================================================
# [[5,1,3]] CODE
# ======================================================================
STABILIZERS = ["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"]
LOGICAL_X = "IIXYX"
LOGICAL_Z = "IIZXZ"
N = 5
PAULIS = ["X", "Y", "Z"]


def pauli_mul_char(a, b):
    if a == "I": return b
    if b == "I": return a
    if a == b: return "I"
    table = {frozenset(["X","Y"]):"Z",
             frozenset(["X","Z"]):"Y",
             frozenset(["Y","Z"]):"X"}
    return table[frozenset([a, b])]


def pauli_mul(p1, p2):
    return "".join(pauli_mul_char(a, b) for a, b in zip(p1, p2))


def pauli_to_symplectic(p):
    n = len(p)
    z = [0] * n
    x = [0] * n
    for i, c in enumerate(p):
        if c == 'X': x[i] = 1
        elif c == 'Z': z[i] = 1
        elif c == 'Y': x[i] = z[i] = 1
    return x + z


def symplectic_product(p1, p2):
    v1 = pauli_to_symplectic(p1)
    v2 = pauli_to_symplectic(p2)
    n = len(p1)
    val = 0
    for i in range(n):
        val += v1[i] * v2[n + i]
        val += v1[n + i] * v2[i]
    return val % 2


def compute_syndrome(error):
    return tuple(symplectic_product(error, s) for s in STABILIZERS)


def build_stab_group():
    g = set()
    for mask in range(16):
        cur = "IIIII"
        for i in range(4):
            if mask & (1 << i):
                cur = pauli_mul(cur, STABILIZERS[i])
        g.add(cur)
    return g


STAB_GROUP = build_stab_group()


def logical_class(residual):
    if residual in STAB_GROUP: return "STABILIZER"
    if pauli_mul(residual, LOGICAL_X) in STAB_GROUP: return "LOGICAL_X"
    if pauli_mul(residual, LOGICAL_Z) in STAB_GROUP: return "LOGICAL_Z"
    ly = pauli_mul(LOGICAL_X, LOGICAL_Z)
    if pauli_mul(residual, ly) in STAB_GROUP: return "LOGICAL_Y"
    return "UNKNOWN"


def build_min_weight_decoder():
    decoder = {}
    for p_tuple in product("IXYZ", repeat=N):
        err = "".join(p_tuple)
        s = compute_syndrome(err)
        w = sum(1 for c in err if c != 'I')
        if s not in decoder or w < decoder[s][1]:
            decoder[s] = (err, w)
    return {s: v[0] for s, v in decoder.items()}


DECODER = build_min_weight_decoder()


def is_logical_failure(error):
    s = compute_syndrome(error)
    correction = DECODER.get(s, "IIIII")
    residual = pauli_mul(error, correction)
    return logical_class(residual) != "STABILIZER"


def cumulative_error(r1, q1, p1, r2, q2, p2):
    err = ["I"] * N
    err[q1] = pauli_mul_char(err[q1], p1)
    err[q2] = pauli_mul_char(err[q2], p2)
    return "".join(err)


def count_same_round(R):
    """Count same-round logical failures."""
    failures = 0
    for r in range(R):
        for q1, q2 in combinations(range(N), 2):
            for p1 in PAULIS:
                for p2 in PAULIS:
                    err = cumulative_error(r, q1, p1, r, q2, p2)
                    if is_logical_failure(err):
                        failures += 1
    return failures


def count_cross_round(R):
    """Count cross-round logical failures."""
    failures = 0
    for r1 in range(R):
        for r2 in range(r1 + 1, R):
            for q1 in range(N):
                for q2 in range(N):
                    for p1 in PAULIS:
                        for p2 in PAULIS:
                            err = cumulative_error(r1, q1, p1, r2, q2, p2)
                            if is_logical_failure(err):
                                failures += 1
    return failures


def main():
    print("=" * 70)
    print("TEST 03: Temporal Two-Fault Scaling — Verify N₂ = 90R²")
    print("=" * 70)

    print(f"\n📊 Code: [[5,1,3]]")
    print(f"   Stabilizers: {STABILIZERS}")
    print(f"   Stabilizer group size: {len(STAB_GROUP)}")

    print(f"\n📌 Article claim:")
    print(f"   N₂^same(D_full; R) = 90R")
    print(f"   N₂^cross(D_full; R) = 90R(R-1)")
    print(f"   N₂(D_full; R) = 90R²")
    print(f"   P_L(p, R) = 10R²·p² + O(p³)")

    print("\n" + "=" * 70)
    print("EXHAUSTIVE ENUMERATION")
    print("=" * 70)

    print(f"\n{'R':>2s} | {'Same':>6s} | {'Cross':>6s} | "
          f"{'Total':>6s} | {'90R²':>6s} | Match?")
    print("-" * 60)

    all_match = True
    for R in [1, 2, 3]:
        same = count_same_round(R)
        cross = count_cross_round(R)
        total = same + cross
        expected = 90 * R * R
        match = (total == expected)
        if not match:
            all_match = False
        symbol = "✅" if match else "❌"
        print(f"{R:>2d} | {same:>6d} | {cross:>6d} | "
              f"{total:>6d} | {expected:>6d} | {symbol}")

    print("\n" + "=" * 70)
    print("VERIFICATION AGAINST FORMULA")
    print("=" * 70)

    for R in [1, 2, 3]:
        same_expected = 90 * R
        cross_expected = 90 * R * (R - 1)
        same = count_same_round(R)
        cross = count_cross_round(R)

        same_match = "✅" if same == same_expected else "❌"
        cross_match = "✅" if cross == cross_expected else "❌"

        print(f"\nR = {R}:")
        print(f"   Same-round:  {same:>4d}  (expected 90R = {same_expected:>4d})  {same_match}")
        print(f"   Cross-round: {cross:>4d}  (expected 90R(R-1) = {cross_expected:>4d})  {cross_match}")

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    print(f"\n📌 Because N₂ = 90R²:")
    print(f"   P_L(p, R) ≈ N₂ × (p/3)²")
    print(f"            = 90R² × p²/9")
    print(f"            = 10R²·p²")
    print(f"   → matches article claim P_L = 10R²·p² + O(p³)")

    print("\n📌 Failure rates:")
    print(f"   Same-round:  100% (90 of 90 per round)")
    print(f"   Cross-round:  80% (180 of 225 per round pair)")

    print("\n" + "=" * 70)
    if all_match:
        print("🎉 TEST 03 PASSED — Article claim verified!")
    else:
        print("❌ TEST 03 FAILED — Article claim INCORRECT!")
    print("=" * 70)


if __name__ == "__main__":
    main()