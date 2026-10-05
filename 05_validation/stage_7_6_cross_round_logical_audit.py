"""
مرحله ۷.۶: آنالیز Logical Failures در چند دور برای [[5,1,3]]
فایل: stage_7_6_cross_round_logical_audit.py
"""

from itertools import product
from collections import defaultdict


# ============================================================
# [[5,1,3]] CODE
# ============================================================

STABILIZERS = [
    "XZZXI",
    "IXZZX",
    "XIXZZ",
    "ZXIXZ",
]

PAULIS = ["X", "Y", "Z"]


# ============================================================
# Pauli utilities
# ============================================================

def pauli_to_symplectic(pauli):

    x = []
    z = []

    for p in pauli:

        if p == "I":
            x.append(0)
            z.append(0)

        elif p == "X":
            x.append(1)
            z.append(0)

        elif p == "Z":
            x.append(0)
            z.append(1)

        elif p == "Y":
            x.append(1)
            z.append(1)

        else:
            raise ValueError(p)

    return x + z


def symplectic_product(p1, p2):

    v1 = pauli_to_symplectic(p1)
    v2 = pauli_to_symplectic(p2)

    n = len(p1)

    x1 = v1[:n]
    z1 = v1[n:]

    x2 = v2[:n]
    z2 = v2[n:]

    value = 0

    for i in range(n):
        value += x1[i] * z2[i]
        value += z1[i] * x2[i]

    return value % 2


def compute_syndrome(error):

    return tuple(
        symplectic_product(error, S)
        for S in STABILIZERS
    )


# ============================================================
# Pauli multiplication
# ============================================================

def pauli_mul_char(a, b):

    if a == "I":
        return b

    if b == "I":
        return a

    if a == b:
        return "I"

    table = {
        frozenset(["X", "Y"]): "Z",
        frozenset(["X", "Z"]): "Y",
        frozenset(["Y", "Z"]): "X",
    }

    return table[frozenset([a, b])]


def pauli_mul(p1, p2):

    return "".join(
        pauli_mul_char(a, b)
        for a, b in zip(p1, p2)
    )


# ============================================================
# Stabilizer group
# ============================================================

def generate_stabilizer_group():

    group = {"IIIII"}

    changed = True

    while changed:

        changed = False

        current = list(group)

        for a in current:
            for b in STABILIZERS:

                c = pauli_mul(a, b)

                if c not in group:
                    group.add(c)
                    changed = True

    return group


STABILIZER_GROUP = generate_stabilizer_group()

print("="*70)
print("مرحله ۷.۶: آنالیز Logical Failures در چند دور")
print("="*70)

print(f"\n📊 اندازه گروه stabilizer: {len(STABILIZER_GROUP)}")
assert len(STABILIZER_GROUP) == 16


# ============================================================
# Logical operators
# ============================================================

LOGICAL_X = "IIXYX"
LOGICAL_Z = "IIYZY"

LOGICAL_Y = pauli_mul(
    LOGICAL_X,
    LOGICAL_Z
)


# ============================================================
# Logical class
# ============================================================

def get_logical_class(residual):

    if residual in STABILIZER_GROUP:
        return "STABILIZER"

    if pauli_mul(residual, LOGICAL_X) in STABILIZER_GROUP:
        return "LOGICAL_X"

    if pauli_mul(residual, LOGICAL_Z) in STABILIZER_GROUP:
        return "LOGICAL_Z"

    if pauli_mul(residual, LOGICAL_Y) in STABILIZER_GROUP:
        return "LOGICAL_Y"

    return "UNKNOWN"


# ============================================================
# Single-round minimum-weight decoder
# ============================================================

def build_decoder():

    decoder = {}

    # Identity
    decoder[(0, 0, 0, 0)] = "IIIII"

    # All single-qubit Pauli errors
    for q in range(5):

        for p in PAULIS:

            error = ["I"] * 5
            error[q] = p

            error = "".join(error)

            syndrome = compute_syndrome(error)

            decoder[syndrome] = error

    assert len(decoder) == 16

    return decoder


DECODER = build_decoder()


# ============================================================
# Cross-round patterns
# ============================================================

def generate_cross_round_patterns(R):

    patterns = []

    for r1 in range(R):

        for r2 in range(r1 + 1, R):

            for q1 in range(5):

                for q2 in range(5):

                    for p1 in PAULIS:

                        for p2 in PAULIS:

                            patterns.append(
                                (
                                    (r1, q1, p1),
                                    (r2, q2, p2)
                                )
                            )

    return patterns


# ============================================================
# Syndrome history
# ============================================================

def compute_syndrome_history(pattern, R):

    cumulative = ["I"] * 5

    history = []

    for r in range(R):

        for fault in pattern:

            fault_round, q, p = fault

            if fault_round == r:

                cumulative[q] = pauli_mul_char(
                    cumulative[q],
                    p
                )

        error = "".join(cumulative)

        history.append(
            compute_syndrome(error)
        )

    return tuple(history)


# ============================================================
# Total error
# ============================================================

def compute_total_error(pattern):

    total = ["I"] * 5

    for r, q, p in pattern:

        total[q] = pauli_mul_char(
            total[q],
            p
        )

    return "".join(total)


# ============================================================
# Cross-round analysis
# ============================================================

def analyze_cross_round(R):

    patterns = generate_cross_round_patterns(R)

    failures = 0

    logical_counts = {
        "STABILIZER": 0,
        "LOGICAL_X": 0,
        "LOGICAL_Y": 0,
        "LOGICAL_Z": 0,
        "UNKNOWN": 0,
    }

    results = []

    for pattern in patterns:

        history = compute_syndrome_history(
            pattern,
            R
        )

        final_syndrome = history[-1]

        correction = DECODER[final_syndrome]

        total_error = compute_total_error(pattern)

        residual = pauli_mul(
            total_error,
            correction
        )

        logical_class = get_logical_class(
            residual
        )

        logical_counts[logical_class] += 1

        if logical_class != "STABILIZER":
            failures += 1

        results.append({
            "pattern": pattern,
            "history": history,
            "total_error": total_error,
            "correction": correction,
            "residual": residual,
            "logical_class": logical_class
        })

    return {
        "R": R,
        "total": len(patterns),
        "failures": failures,
        "success": len(patterns) - failures,
        "logical_counts": logical_counts,
        "results": results
    }


# ============================================================
# Run
# ============================================================

R_values = [2, 3, 4, 5, 10]

all_results = {}

for R in R_values:

    result = analyze_cross_round(R)

    all_results[R] = result

    print()
    print(f"R = {R}")
    print("-" * 50)

    print(
        "Total cross-round patterns =",
        result["total"]
    )

    print(
        "Logical failures =",
        result["failures"]
    )

    print(
        "Success =",
        result["success"]
    )

    print()
    print("Logical class distribution:")

    for cls, count in result["logical_counts"].items():

        print(
            f"   {cls:12s}: {count}"
        )


# ============================================================
# Final summary
# ============================================================

print()
print("="*70)
print("📊 خلاصه نهایی")
print("="*70)

for R, result in all_results.items():

    print(
        f"R={R:2d} | "
        f"Total={result['total']:5d} | "
        f"Failures={result['failures']:5d} | "
        f"Success={result['success']:5d} | "
        f"Success Rate={result['success']/result['total']*100:.2f}%"
    )

print("\n" + "="*70)
print("📌 نتیجه:")
print("="*70)
print("   ✅ Logical failures در چند دور آنالیز شد.")
print("   ✅ نرخ موفقیت Decoder محاسبه شد.")
print("   ✅ توزیع Logical classes بررسی شد.")
print("="*70)

print("\n✅ مرحله ۷.۶ کامل شد.")