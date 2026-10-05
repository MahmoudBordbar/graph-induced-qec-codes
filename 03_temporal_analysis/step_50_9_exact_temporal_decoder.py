"""
======================================================================
مرحله 50.9: Exact Temporal Syndrome-History Decoder
[[5,1,3]]

نسخه اصلاح‌شده

اصلاح اصلی:
    یک Pauli با syndrome غیرصفر مستقیماً logical class ندارد.

ابتدا:
    syndrome(E)
    -> minimum-weight correction C_s
    -> residual = E*C_s
    -> logical coset(residual)

در نتیجه:
    هر Pauli به یکی از 64 کلاس

        syndrome × logical class

نگاشت می‌شود.

هدف:
    R=1 باید دقیقاً با Stage 28 منطبق شود.

======================================================================
"""

import numpy as np
from itertools import product
from collections import defaultdict


# ======================================================================
# 1. CANONICAL [[5,1,3]]
# ======================================================================

STABILIZERS = [
    "XZZXI",
    "IXZZX",
    "XIXZZ",
    "ZXIXZ",
]

LOGICAL_X = "IIXYX"
LOGICAL_Z = "IIZXZ"

N = 5

LOGICAL_CLASSES = [
    "stabilizer",
    "logical_X",
    "logical_Z",
    "logical_Y",
]


# ======================================================================
# 2. EXACT SINGLE-SHOT P_L
# ======================================================================

def pl_exact_single_shot(p):
    return (
        90 * (1 - p)**3 * (p / 3)**2
        + 210 * (1 - p)**2 * (p / 3)**3
        + 270 * (1 - p) * (p / 3)**4
        + 198 * (p / 3)**5
    )


# ======================================================================
# 3. PAULI MULTIPLICATION
# ======================================================================

PAULI_MUL_TABLE = {
    ("I", "I"): "I",
    ("I", "X"): "X",
    ("I", "Y"): "Y",
    ("I", "Z"): "Z",

    ("X", "I"): "X",
    ("Y", "I"): "Y",
    ("Z", "I"): "Z",

    ("X", "X"): "I",
    ("Y", "Y"): "I",
    ("Z", "Z"): "I",

    ("X", "Y"): "Z",
    ("Y", "X"): "Z",

    ("Y", "Z"): "X",
    ("Z", "Y"): "X",

    ("Z", "X"): "Y",
    ("X", "Z"): "Y",
}


def pauli_mul_string(p1, p2):

    return "".join(
        PAULI_MUL_TABLE[(a, b)]
        for a, b in zip(p1, p2)
    )


# ======================================================================
# 4. SYMPLECTIC ALGEBRA
# ======================================================================

def pauli_to_symplectic(pauli):

    z = np.zeros(N, dtype=np.int8)
    x = np.zeros(N, dtype=np.int8)

    for i, p in enumerate(pauli):

        if p == "X":
            x[i] = 1

        elif p == "Z":
            z[i] = 1

        elif p == "Y":
            x[i] = 1
            z[i] = 1

    return np.concatenate([z, x])


def symplectic_product(p1, p2):

    v1 = pauli_to_symplectic(p1)
    v2 = pauli_to_symplectic(p2)

    z1 = v1[:N]
    x1 = v1[N:]

    z2 = v2[:N]
    x2 = v2[N:]

    return int(
        (
            np.dot(z1, x2)
            + np.dot(x1, z2)
        ) % 2
    )


# ======================================================================
# 5. SYNDROME
# ======================================================================

def syndrome_of(error):

    return tuple(
        symplectic_product(
            error,
            stab
        )
        for stab in STABILIZERS
    )


# ======================================================================
# 6. STABILIZER GROUP
# ======================================================================

def build_stabilizer_group():

    group = set()

    for mask in range(16):

        current = "IIIII"

        for i, generator in enumerate(
            STABILIZERS
        ):

            if mask & (1 << i):

                current = pauli_mul_string(
                    current,
                    generator
                )

        group.add(current)

    return group


STABILIZER_GROUP = build_stabilizer_group()


# ======================================================================
# 7. BUILD MINIMUM-WEIGHT CORRECTION FOR EACH SYNDROME
# ======================================================================

def build_syndrome_decoder():

    decoder = {}

    for error in product(
        ["I", "X", "Y", "Z"],
        repeat=N
    ):

        error = "".join(error)

        syndrome = syndrome_of(
            error
        )

        weight = sum(
            c != "I"
            for c in error
        )

        if syndrome not in decoder:

            decoder[syndrome] = (
                error,
                weight
            )

        elif weight < decoder[syndrome][1]:

            decoder[syndrome] = (
                error,
                weight
            )

    return {
        syndrome: correction[0]
        for syndrome, correction in decoder.items()
    }


DECODER = build_syndrome_decoder()


# ======================================================================
# 8. LOGICAL COSSET
# ======================================================================

def logical_class_from_normalizer(
    residual
):
    """
    residual MUST have zero syndrome.
    """

    if residual not in STABILIZER_GROUP:

        x_coset = pauli_mul_string(
            residual,
            LOGICAL_X
        )

        if x_coset in STABILIZER_GROUP:
            return "logical_X"

        z_coset = pauli_mul_string(
            residual,
            LOGICAL_Z
        )

        if z_coset in STABILIZER_GROUP:
            return "logical_Z"

        logical_y = pauli_mul_string(
            LOGICAL_X,
            LOGICAL_Z
        )

        y_coset = pauli_mul_string(
            residual,
            logical_y
        )

        if y_coset in STABILIZER_GROUP:
            return "logical_Y"

    else:
        return "stabilizer"

    raise RuntimeError(
        f"Normalizer Pauli has unknown logical class: {residual}"
    )


# ======================================================================
# 9. CORRECTED LOGICAL CLASS OF ARBITRARY PAULI
# ======================================================================

def corrected_logical_class(error):

    syndrome = syndrome_of(
        error
    )

    correction = DECODER[
        syndrome
    ]

    residual = pauli_mul_string(
        error,
        correction
    )

    # Must now commute with all stabilizers.
    if syndrome_of(residual) != (
        0,
        0,
        0,
        0
    ):

        raise RuntimeError(
            "Correction failed to remove syndrome."
        )

    return logical_class_from_normalizer(
        residual
    )


# ======================================================================
# 10. ENUMERATE 1024 PAULIS
# ======================================================================

ALL_PAULIS = [
    "".join(p)
    for p in product(
        ["I", "X", "Y", "Z"],
        repeat=N
    )
]


# ======================================================================
# 11. BUILD 64 CLASSES
# ======================================================================

def build_quotient_classes():

    classes = {}

    for error in ALL_PAULIS:

        syndrome = syndrome_of(
            error
        )

        logical = corrected_logical_class(
            error
        )

        key = (
            syndrome,
            logical
        )

        if key not in classes:

            classes[key] = error

    return classes


QUOTIENT_REPRESENTATIVES = (
    build_quotient_classes()
)


# ======================================================================
# 12. VERIFY 64 CLASSES
# ======================================================================

def verify_quotient_classes():

    expected = 16 * 4
    actual = len(
        QUOTIENT_REPRESENTATIVES
    )

    print(
        f"Expected quotient classes: {expected}"
    )

    print(
        f"Actual quotient classes:   {actual}"
    )

    if actual != expected:

        raise RuntimeError(
            "Quotient-class construction failed."
        )

    print(
        "✅ 64 = 16 syndrome classes × 4 logical classes"
    )


# ======================================================================
# 13. STABLE CLASS ORDER
# ======================================================================

CLASS_KEYS = sorted(
    QUOTIENT_REPRESENTATIVES.keys(),
    key=lambda item: (
        item[0],
        item[1]
    )
)

CLASS_ID = {
    key: i
    for i, key in enumerate(
        CLASS_KEYS
    )
}

NUM_CLASSES = len(
    CLASS_KEYS
)


# ======================================================================
# 14. PAULI -> CLASS
# ======================================================================

PAULI_TO_CLASS = {}

for error in ALL_PAULIS:

    key = (
        syndrome_of(error),
        corrected_logical_class(error)
    )

    PAULI_TO_CLASS[error] = CLASS_ID[
        key
    ]


# ======================================================================
# 15. PHYSICAL ERROR DISTRIBUTION
# ======================================================================

def error_class_distribution(p):

    probabilities = np.zeros(
        NUM_CLASSES,
        dtype=np.float64
    )

    for error in ALL_PAULIS:

        weight = sum(
            c != "I"
            for c in error
        )

        prob = (
            (1 - p) ** (N - weight)
            * (p / 3) ** weight
        )

        class_id = PAULI_TO_CLASS[
            error
        ]

        probabilities[class_id] += prob

    if not np.isclose(
        probabilities.sum(),
        1.0,
        atol=1e-12
    ):

        raise RuntimeError(
            "Error-class probabilities do not sum to 1."
        )

    return probabilities


# ======================================================================
# 16. TRANSITION MATRIX
# ======================================================================

def build_transition_matrix(p):

    """
    A class representative represents an accumulated Pauli class.

    A fresh physical Pauli fault is sampled from the full Pauli channel.

    The new class is obtained after multiplying both and applying
    the minimum-weight syndrome correction only for classification.
    """

    T = np.zeros(
        (NUM_CLASSES, NUM_CLASSES),
        dtype=np.float64
    )

    fault_probabilities = (
        error_class_distribution(p)
    )

    representatives = [
        QUOTIENT_REPRESENTATIVES[key]
        for key in CLASS_KEYS
    ]

    for current_id, current_error in enumerate(
        representatives
    ):

        for fault_id, fault_probability in enumerate(
            fault_probabilities
        ):

            if fault_probability == 0:
                continue

            fault_error = (
                QUOTIENT_REPRESENTATIVES[
                    CLASS_KEYS[fault_id]
                ]
            )

            combined = pauli_mul_string(
                current_error,
                fault_error
            )

            syndrome = syndrome_of(
                combined
            )

            logical = corrected_logical_class(
                combined
            )

            new_key = (
                syndrome,
                logical
            )

            new_id = CLASS_ID[
                new_key
            ]

            T[
                current_id,
                new_id
            ] += fault_probability

    if not np.allclose(
        T.sum(axis=1),
        1.0,
        atol=1e-12
    ):

        raise RuntimeError(
            "Transition matrix rows do not sum to 1."
        )

    return T


# ======================================================================
# 17. EXACT TEMPORAL DECODER
# ======================================================================

def exact_temporal_pl(
    p,
    rounds
):
    """
    Bayesian decoder using the full syndrome history.

    State:
        64 quotient classes

    Observation at each round:
        4-bit syndrome

    At the end:
        choose the most probable logical coset.
    """

    T = build_transition_matrix(
        p
    )

    identity_key = (
        (0, 0, 0, 0),
        "stabilizer"
    )

    identity_id = CLASS_ID[
        identity_key
    ]

    # history -> probability vector
    histories = {
        (): np.eye(
            NUM_CLASSES
        )[identity_id]
    }

    for _ in range(rounds):

        new_histories = {}

        for history, state in histories.items():

            transitioned = (
                state @ T
            )

            # There are 16 possible syndrome observations.
            for syndrome in product(
                [0, 1],
                repeat=4
            ):

                mask = np.array(
                    [
                        key[0] == syndrome
                        for key in CLASS_KEYS
                    ],
                    dtype=bool
                )

                branch = transitioned.copy()

                branch[~mask] = 0.0

                probability = (
                    branch.sum()
                )

                if probability == 0:
                    continue

                new_history = (
                    history
                    + (syndrome,)
                )

                new_histories[
                    new_history
                ] = branch

        histories = new_histories

    total = 0.0
    success = 0.0

    logical_distribution = defaultdict(
        float
    )

    for state in histories.values():

        probability = state.sum()

        if probability <= 0:
            continue

        total += probability

        masses = {
            logical: 0.0
            for logical in LOGICAL_CLASSES
        }

        for class_id, key in enumerate(
            CLASS_KEYS
        ):

            masses[
                key[1]
            ] += state[class_id]

        best = max(
            masses.values()
        )

        success += best

        for logical in LOGICAL_CLASSES:

            logical_distribution[
                logical
            ] += masses[logical]

    if not np.isclose(
        total,
        1.0,
        atol=1e-10
    ):

        raise RuntimeError(
            f"Temporal probability mass = {total}"
        )

    pl = total - success

    return pl, logical_distribution


# ======================================================================
# 18. TEST R=1
# ======================================================================

def test_round_1():

    print("\n" + "=" * 70)
    print(
        "TEST 1: R=1 Exact Validation"
    )
    print("=" * 70)

    test_points = [
        0.001,
        0.005,
        0.01,
        0.02,
        0.05,
        0.10,
    ]

    passed = True

    for p in test_points:

        temporal, _ = (
            exact_temporal_pl(
                p,
                rounds=1
            )
        )

        reference = (
            pl_exact_single_shot(p)
        )

        diff = abs(
            temporal - reference
        )

        print(
            f"p={p:.3f} | "
            f"temporal={temporal:.10f} | "
            f"reference={reference:.10f} | "
            f"diff={diff:.3e}"
        )

        if not np.isclose(
            temporal,
            reference,
            atol=1e-12
        ):

            passed = False

    if passed:

        print(
            "\n✅ R=1 EXACTLY matches Stage 28"
        )

        return True

    print(
        "\n❌ R=1 mismatch"
    )

    return False


# ======================================================================
# 19. TEMPORAL ANALYSIS
# ======================================================================

def run_temporal_analysis():

    print("\n" + "=" * 70)
    print(
        "TEST 2: Temporal Decoder"
    )
    print("=" * 70)

    p_values = [
        0.001,
        0.002,
        0.005,
        0.01,
        0.02,
        0.05,
    ]

    for rounds in [1, 2, 3]:

        print(
            f"\nRounds = {rounds}"
        )

        print(
            "p        | P_L            | P_L/p²"
        )

        print(
            "---------|-----------------|-----------"
        )

        for p in p_values:

            pl, _ = (
                exact_temporal_pl(
                    p,
                    rounds
                )
            )

            ratio = (
                pl / p**2
            )

            print(
                f"{p:.3f}    | "
                f"{pl:.12f} | "
                f"{ratio:.6f}"
            )


# ======================================================================
# 20. LOGICAL DISTRIBUTION
# ======================================================================

def print_logical_distribution():

    print("\n" + "=" * 70)
    print(
        "TEST 3: Logical Distribution"
    )
    print("=" * 70)

    p = 0.01
    rounds = 3

    pl, distribution = (
        exact_temporal_pl(
            p,
            rounds
        )
    )

    print(
        f"p = {p}"
    )

    print(
        f"rounds = {rounds}"
    )

    print(
        f"P_L = {pl:.12f}"
    )

    print(
        "\nLogical class distribution:"
    )

    for logical in LOGICAL_CLASSES:

        print(
            f"   {logical:12s}: "
            f"{distribution[logical]:.12f}"
        )


# ======================================================================
# 21. MAIN
# ======================================================================

def run_stage_50_9():

    print("=" * 70)
    print(
        "مرحله 50.9: Exact Temporal Syndrome-History Decoder"
    )
    print(
        "[[5,1,3]]"
    )
    print("=" * 70)

    print(
        "\nState space:"
    )

    print(
        "   Physical Pauli operators = 1024"
    )

    print(
        "   Stabilizer group size     = 16"
    )

    print(
        "   Quotient classes          = 64"
    )

    print(
        "   Syndrome classes          = 16"
    )

    print(
        "   Logical classes           = 4"
    )

    # ----------------------------------------------------------
    # Verify decoder
    # ----------------------------------------------------------

    verify_quotient_classes()

    # ----------------------------------------------------------
    # Verify R=1
    # ----------------------------------------------------------

    if not test_round_1():

        print(
            "\n❌ STAGE 50.9 FAILED at TEST 1"
        )

        return

    # ----------------------------------------------------------
    # Temporal analysis
    # ----------------------------------------------------------

    run_temporal_analysis()

    # ----------------------------------------------------------
    # Logical distribution
    # ----------------------------------------------------------

    print_logical_distribution()

    print("\n" + "=" * 70)
    print(
        "🏆 STAGE 50.9 COMPLETE"
    )
    print("=" * 70)

    print("""
✅ Corrected arbitrary-Pauli logical classification
✅ 64 exact syndrome × logical quotient classes
✅ R=1 cross-check against Stage 28
✅ Exact temporal decoder constructed
✅ No graphlike approximation
✅ No PyMatching/MWPM approximation
""")


if __name__ == "__main__":
    run_stage_50_9()