# ============================================================
# FILE NAME: full_history_collision_audit.py
# PROJECT: [[5,1,3]] Quantum Error Correction Audit
# PURPOSE:
# Full syndrome-history collision analysis for
# single, same-round, and cross-round fault patterns.
# ============================================================

from itertools import combinations
from collections import defaultdict


# ============================================================
# [[5,1,3]] FIVE-QUBIT CODE
# ============================================================

STABILIZERS = [
    "XZZXI",
    "IXZZX",
    "XIXZZ",
    "ZXIXZ",
]

PAULIS = ["X", "Y", "Z"]
IDENTITY = "IIIII"


# ============================================================
# PAULI MULTIPLICATION
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

    assert len(p1) == len(p2)

    return "".join(
        pauli_mul_char(a, b)
        for a, b in zip(p1, p2)
    )


# ============================================================
# SYMPLECTIC REPRESENTATION
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
            raise ValueError(f"Invalid Pauli: {p}")

    return x + z


def symplectic_product(p1, p2):

    v1 = pauli_to_symplectic(p1)
    v2 = pauli_to_symplectic(p2)

    n = len(p1)

    value = 0

    for i in range(n):
        value += v1[i] * v2[n + i]
        value += v1[n + i] * v2[i]

    return value % 2


# ============================================================
# SYNDROME
# ============================================================

def compute_syndrome(error):

    return tuple(
        symplectic_product(error, stabilizer)
        for stabilizer in STABILIZERS
    )


# ============================================================
# STABILIZER GROUP
# ============================================================

def generate_stabilizer_group():

    group = set()

    for mask in range(16):

        current = IDENTITY

        for i in range(4):

            if (mask >> i) & 1:

                current = pauli_mul(
                    current,
                    STABILIZERS[i]
                )

        group.add(current)

    return group


STABILIZER_GROUP = generate_stabilizer_group()


# ============================================================
# LOGICAL OPERATORS
# ============================================================

LOGICAL_X = "IIXYX"
LOGICAL_Z = "IIZXZ"

LOGICAL_Y = pauli_mul(
    LOGICAL_X,
    LOGICAL_Z
)


def get_logical_class(residual):

    if residual in STABILIZER_GROUP:
        return "STABILIZER"

    if pauli_mul(
        residual,
        LOGICAL_X
    ) in STABILIZER_GROUP:
        return "LOGICAL_X"

    if pauli_mul(
        residual,
        LOGICAL_Y
    ) in STABILIZER_GROUP:
        return "LOGICAL_Y"

    if pauli_mul(
        residual,
        LOGICAL_Z
    ) in STABILIZER_GROUP:
        return "LOGICAL_Z"

    return "UNKNOWN"


# ============================================================
# SINGLE-FAULT PATTERNS
# ============================================================

def generate_single_fault_patterns(R):

    patterns = []

    for r in range(R):

        for q in range(5):

            for p in PAULIS:

                patterns.append({
                    "type": "single",
                    "faults": [
                        (r, q, p)
                    ]
                })

    return patterns


# ============================================================
# SAME-ROUND TWO-FAULT PATTERNS
# ============================================================

def generate_same_round_patterns(R):

    patterns = []

    for r in range(R):

        for q1, q2 in combinations(range(5), 2):

            for p1 in PAULIS:

                for p2 in PAULIS:

                    patterns.append({
                        "type": "same_round",
                        "faults": [
                            (r, q1, p1),
                            (r, q2, p2)
                        ]
                    })

    return patterns


# ============================================================
# CROSS-ROUND TWO-FAULT PATTERNS
# ============================================================

def generate_cross_round_patterns(R):

    patterns = []

    for r1 in range(R):

        for r2 in range(r1 + 1, R):

            for q1 in range(5):

                for q2 in range(5):

                    for p1 in PAULIS:

                        for p2 in PAULIS:

                            patterns.append({
                                "type": "cross_round",
                                "faults": [
                                    (r1, q1, p1),
                                    (r2, q2, p2)
                                ]
                            })

    return patterns


# ============================================================
# SYNDROME HISTORY
# ============================================================

def syndrome_history(pattern, R):

    cumulative = ["I"] * 5

    history = []

    for r in range(R):

        for fault_round, q, p in pattern["faults"]:

            if fault_round == r:

                cumulative[q] = pauli_mul_char(
                    cumulative[q],
                    p
                )

        current_error = "".join(cumulative)

        syndrome = compute_syndrome(
            current_error
        )

        history.append(syndrome)

    return tuple(history)


# ============================================================
# FINAL CUMULATIVE ERROR
# ============================================================

def total_error(pattern):

    cumulative = ["I"] * 5

    for _, q, p in pattern["faults"]:

        cumulative[q] = pauli_mul_char(
            cumulative[q],
            p
        )

    return "".join(cumulative)


# ============================================================
# HISTORY COLLISION ANALYSIS
# ============================================================

def analyze_history_collisions(R):

    print("\n")
    print("=" * 70)
    print("FULL SYNDROME-HISTORY COLLISION AUDIT")
    print("=" * 70)

    single = generate_single_fault_patterns(R)

    same = generate_same_round_patterns(R)

    cross = generate_cross_round_patterns(R)

    all_patterns = (
        single +
        same +
        cross
    )

    history_map = defaultdict(list)

    for pattern in all_patterns:

        H = syndrome_history(
            pattern,
            R
        )

        history_map[H].append(
            pattern
        )

    unique_histories = len(history_map)

    ambiguous = {
        H: patterns
        for H, patterns in history_map.items()
        if len(patterns) > 1
    }

    print(f"\nR = {R}")
    print("-" * 50)

    print(
        "Single-fault patterns        =",
        len(single)
    )

    print(
        "Same-round 2-fault patterns  =",
        len(same)
    )

    print(
        "Cross-round 2-fault patterns =",
        len(cross)
    )

    print(
        "Total patterns               =",
        len(all_patterns)
    )

    print(
        "Unique histories             =",
        unique_histories
    )

    print(
        "Ambiguous histories          =",
        len(ambiguous)
    )

    if ambiguous:

        max_multiplicity = max(
            len(patterns)
            for patterns in ambiguous.values()
        )

    else:

        max_multiplicity = 1

    print(
        "Maximum multiplicity          =",
        max_multiplicity
    )

    # --------------------------------------------------------
    # Collision categories
    # --------------------------------------------------------

    single_vs_same = 0
    single_vs_cross = 0
    same_vs_cross = 0
    cross_vs_cross = 0

    for H, patterns in ambiguous.items():

        types = [
            p["type"]
            for p in patterns
        ]

        type_set = set(types)

        if (
            "single" in type_set
            and "same_round" in type_set
        ):
            single_vs_same += 1

        if (
            "single" in type_set
            and "cross_round" in type_set
        ):
            single_vs_cross += 1

        if (
            "same_round" in type_set
            and "cross_round" in type_set
        ):
            same_vs_cross += 1

        if types.count("cross_round") >= 2:
            cross_vs_cross += 1

    print("\nCollision categories:")
    print(
        "Single vs Same-round        =",
        single_vs_same
    )

    print(
        "Single vs Cross-round       =",
        single_vs_cross
    )

    print(
        "Same-round vs Cross-round   =",
        same_vs_cross
    )

    print(
        "Cross-round vs Cross-round  =",
        cross_vs_cross
    )

    # --------------------------------------------------------
    # Examples
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("EXAMPLE AMBIGUOUS HISTORIES")
    print("=" * 70)

    if not ambiguous:

        print(
            "No ambiguous syndrome histories found."
        )

    else:

        shown = 0

        for H, patterns in ambiguous.items():

            print("\nHistory:")
            print(H)

            for pattern in patterns[:10]:

                print(
                    "   Type:",
                    pattern["type"],
                    "| Faults:",
                    pattern["faults"]
                )

            shown += 1

            if shown >= 10:
                break

    return {
        "total": len(all_patterns),
        "unique": unique_histories,
        "ambiguous": len(ambiguous),
        "single_vs_same": single_vs_same,
        "single_vs_cross": single_vs_cross,
        "same_vs_cross": same_vs_cross,
        "cross_vs_cross": cross_vs_cross,
    }


# ============================================================
# CROSS-ROUND RAW ERROR ANALYSIS
# ============================================================

def analyze_cross_round_errors(R):

    patterns = generate_cross_round_patterns(R)

    classes = defaultdict(int)

    for pattern in patterns:

        error = total_error(pattern)

        logical_class = get_logical_class(
            error
        )

        classes[logical_class] += 1

    print("\n")
    print("=" * 70)
    print("CROSS-ROUND RAW ERROR CLASS AUDIT")
    print("=" * 70)

    print(f"\nR = {R}")
    print("-" * 50)

    print(
        "Total cross-round patterns =",
        len(patterns)
    )

    for key in [
        "STABILIZER",
        "LOGICAL_X",
        "LOGICAL_Y",
        "LOGICAL_Z",
        "UNKNOWN"
    ]:

        print(
            f"{key:15s}:",
            classes[key]
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("FILE: full_history_collision_audit.py")
    print("[[5,1,3]] FULL SYNDROME-HISTORY AUDIT")
    print("=" * 70)

    print(
        "\nStabilizer group size:",
        len(STABILIZER_GROUP)
    )

    R_VALUES = [2, 3, 4, 5, 10]

    results = {}

    for R in R_VALUES:

        results[R] = analyze_history_collisions(R)

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    for R, result in results.items():

        print(
            f"R={R:2d} | "
            f"Total={result['total']:5d} | "
            f"Unique={result['unique']:5d} | "
            f"Ambiguous={result['ambiguous']:4d}"
        )

        print(
            f"      "
            f"Single/Same={result['single_vs_same']} | "
            f"Single/Cross={result['single_vs_cross']} | "
            f"Same/Cross={result['same_vs_cross']} | "
            f"Cross/Cross={result['cross_vs_cross']}"
        )

    print("\n")
    print("=" * 70)
    print("AUDIT COMPLETED")
    print("=" * 70)