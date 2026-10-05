"""
======================================================================
مرحله 51: Efficient Exact Temporal Decoder + Round Scaling
[[5,1,3]]

هدف:
    محاسبه دقیق P_L(p, R) برای تعداد roundهای بیشتر بدون
    enumerate کردن مستقیم 16^R syndrome histories.

ایده:
    در round r:

        s_r = syndrome(E_1 E_2 ... E_r)

    بنابراین:

        delta_s = s_r XOR s_{r-1}

    دقیقاً syndrome خطای فیزیکی round جاری است.

برای هر syndrome delta، توزیع conditional چهار logical class
را از تمام 1024 Pauli errors استخراج می‌کنیم.

سپس decoder با belief-state dynamic programming پیش می‌رود.

ویژگی‌ها:
    - بدون PyMatching
    - بدون graph decomposition
    - بدون DEM approximation
    - exact Pauli channel
    - exact syndrome information
    - efficient temporal inference

اعتبارسنجی:
    R=1 باید دقیقاً با Stage 28 برابر باشد.
    R=2,3 باید Stage 50.9 را بازتولید کنند.

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

# Logical Pauli multiplication.
#
# stabilizer = 0
# logical_X  = 1
# logical_Z  = 2
# logical_Y  = 3
#
# The logical Pauli group modulo phase is Z2 x Z2.
LOGICAL_INDEX = {
    "stabilizer": 0,
    "logical_X": 1,
    "logical_Z": 2,
    "logical_Y": 3,
}


# ======================================================================
# 2. EXACT SINGLE-SHOT P_L
# ======================================================================

def pl_exact_single_shot(p):
    """
    Exact P_L from Stage 28.
    """

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
    """Pauli multiplication ignoring global phase."""

    try:
        return "".join(
            PAULI_MUL_TABLE[(a, b)]
            for a, b in zip(p1, p2)
        )
    except KeyError as exc:
        raise ValueError(
            f"Invalid Pauli multiplication for {p1} and {p2}"
        ) from exc


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

        elif p != "I":
            raise ValueError(
                f"Invalid Pauli character: {p}"
            )

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
            stabilizer
        )
        for stabilizer in STABILIZERS
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
# 7. MINIMUM-WEIGHT DECODER
# ======================================================================

def build_syndrome_decoder():

    decoder = {}

    for pauli_tuple in product(
        ["I", "X", "Y", "Z"],
        repeat=N
    ):

        error = "".join(
            pauli_tuple
        )

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
        syndrome: data[0]
        for syndrome, data in decoder.items()
    }


DECODER = build_syndrome_decoder()


# ======================================================================
# 8. LOGICAL CLASS OF NORMALIZER ELEMENT
# ======================================================================

def logical_class_from_normalizer(
    residual
):
    """
    residual must commute with all stabilizers.
    """

    if residual in STABILIZER_GROUP:

        return "stabilizer"

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

    raise RuntimeError(
        f"Unknown logical normalizer element: {residual}"
    )


# ======================================================================
# 9. CLASSIFICATION OF ARBITRARY PAULI
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

    if syndrome_of(residual) != (
        0,
        0,
        0,
        0
    ):

        raise RuntimeError(
            "Minimum-weight correction did not remove syndrome."
        )

    return logical_class_from_normalizer(
        residual
    )


# ======================================================================
# 10. ALL 1024 PAULIS
# ======================================================================

ALL_PAULIS = [
    "".join(p)
    for p in product(
        ["I", "X", "Y", "Z"],
        repeat=N
    )
]


# ======================================================================
# 11. BUILD CONDITIONAL FAULT DISTRIBUTIONS
# ======================================================================

def build_syndrome_logical_table(p):
    """
    For each physical-fault syndrome s:

        q[s] = P(syndrome = s)

    and

        K[s][logical] =
            P(logical class | syndrome=s)

    These are obtained exactly by enumeration of all 1024 Pauli errors.
    """

    joint = {
        syndrome: np.zeros(
            4,
            dtype=np.float64
        )
        for syndrome in product(
            [0, 1],
            repeat=4
        )
    }

    for error in ALL_PAULIS:

        weight = sum(
            c != "I"
            for c in error
        )

        probability = (
            (1 - p) ** (N - weight)
            * (p / 3) ** weight
        )

        syndrome = syndrome_of(
            error
        )

        logical = corrected_logical_class(
            error
        )

        logical_id = LOGICAL_INDEX[
            logical
        ]

        joint[
            syndrome
        ][logical_id] += probability

    syndrome_probability = {}

    conditional_logical = {}

    for syndrome, values in joint.items():

        total = values.sum()

        syndrome_probability[
            syndrome
        ] = total

        if total > 0:

            conditional_logical[
                syndrome
            ] = values / total

        else:

            conditional_logical[
                syndrome
            ] = np.zeros(
                4,
                dtype=np.float64
            )

    # Sanity checks.
    total_probability = sum(
        syndrome_probability.values()
    )

    if not np.isclose(
        total_probability,
        1.0,
        atol=1e-12
    ):

        raise RuntimeError(
            f"Total syndrome probability = "
            f"{total_probability}"
        )

    return (
        syndrome_probability,
        conditional_logical
    )


# ======================================================================
# 12. LOGICAL CONVOLUTION
# ======================================================================

def logical_convolution(
    belief,
    fault_distribution
):
    """
    Logical class multiplication:

        I = 0
        X = 1
        Z = 2
        Y = 3

    Because the logical Pauli group modulo phase is Z2 x Z2,
    multiplication corresponds to XOR of indices.
    """

    new_belief = np.zeros(
        4,
        dtype=np.float64
    )

    for current in range(4):

        if belief[current] == 0:
            continue

        for fault in range(4):

            if fault_distribution[fault] == 0:
                continue

            new_class = current ^ fault

            new_belief[new_class] += (
                belief[current]
                * fault_distribution[fault]
            )

    return new_belief


# ======================================================================
# 13. BELIEF-STATE KEY
# ======================================================================

def belief_key(
    belief,
    decimals=14
):
    """
    Numerical key for belief-state memoization.

    Rounding is used only for hashing computationally equivalent
    floating-point belief states.
    """

    rounded = np.round(
        belief,
        decimals
    )

    rounded[
        np.abs(rounded) < 1e-15
    ] = 0.0

    return tuple(
        float(x)
        for x in rounded
    )


# ======================================================================
# 14. EXACT TEMPORAL DECODER
# ======================================================================

def exact_temporal_decoder(
    p,
    rounds,
    return_state_statistics=False
):
    """
    Exact Bayesian temporal decoder.

    State:
        posterior probability over 4 logical classes.

    At each round:
        1. sample/observe the physical fault syndrome
        2. update logical belief using the conditional
           logical distribution for that syndrome.

    Instead of explicitly storing 16^R histories, identical
    belief states are merged.

    Returns optimal logical failure probability.
    """

    (
        syndrome_probability,
        conditional_logical,
    ) = build_syndrome_logical_table(
        p
    )

    # ----------------------------------------------------------
    # Start:
    # prior logical class = identity.
    # ----------------------------------------------------------

    initial_belief = np.array(
        [1.0, 0.0, 0.0, 0.0],
        dtype=np.float64
    )

    # Mapping:
    # belief_key -> probability of reaching that belief state.
    states = {
        belief_key(
            initial_belief
        ): 1.0
    }

    for round_index in range(
        rounds
    ):

        new_states = defaultdict(
            float
        )

        for key, history_probability in (
            states.items()
        ):

            belief = np.array(
                key,
                dtype=np.float64
            )

            for syndrome in syndrome_probability:

                p_s = syndrome_probability[
                    syndrome
                ]

                if p_s <= 0:
                    continue

                kernel = conditional_logical[
                    syndrome
                ]

                new_belief = (
                    logical_convolution(
                        belief,
                        kernel
                    )
                )

                new_key = belief_key(
                    new_belief
                )

                new_states[
                    new_key
                ] += (
                    history_probability
                    * p_s
                )

        states = dict(
            new_states
        )

        if not states:

            raise RuntimeError(
                "Temporal decoder lost all probability mass."
            )

    # ----------------------------------------------------------
    # Optimal final logical decision.
    # ----------------------------------------------------------

    total_probability = 0.0
    total_success = 0.0

    for key, history_probability in (
        states.items()
    ):

        belief = np.array(
            key,
            dtype=np.float64
        )

        total_probability += (
            history_probability
        )

        # MAP logical decision.
        best_probability = np.max(
            belief
        )

        total_success += (
            history_probability
            * best_probability
        )

    if not np.isclose(
        total_probability,
        1.0,
        atol=1e-10
    ):

        raise RuntimeError(
            f"Final probability mass = "
            f"{total_probability}"
        )

    pl = (
        1.0
        - total_success
    )

    if return_state_statistics:

        return (
            pl,
            len(states),
            total_probability,
        )

    return pl


# ======================================================================
# 15. TEST R=1
# ======================================================================

def test_r1():

    print("\n" + "=" * 70)
    print(
        "TEST 1: R=1 Exact Cross-Validation"
    )
    print("=" * 70)

    points = [
        0.001,
        0.005,
        0.01,
        0.02,
        0.05,
        0.10,
    ]

    passed = True

    for p in points:

        temporal = exact_temporal_decoder(
            p,
            rounds=1
        )

        reference = (
            pl_exact_single_shot(
                p
            )
        )

        difference = abs(
            temporal
            - reference
        )

        print(
            f"p={p:.3f} | "
            f"temporal={temporal:.12f} | "
            f"reference={reference:.12f} | "
            f"diff={difference:.3e}"
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
# 16. VALIDATE R=2,3 AGAINST STAGE 50.9
# ======================================================================

def test_r2_r3():

    print("\n" + "=" * 70)
    print(
        "TEST 2: R=2/R=3 Cross-Validation"
    )
    print("=" * 70)

    # Expected values from Stage 50.9.
    expected = {
        2: {
            0.001: 0.000019955458,
            0.002: 0.000079642899,
            0.005: 0.000494385138,
            0.010: 0.001954634968,
        },

        3: {
            0.001: 0.000029932989,
            0.002: 0.000119461177,
            0.005: 0.000741455485,
            0.010: 0.002930041323,
        },
    }

    passed = True

    for rounds in [2, 3]:

        print(
            f"\nRounds = {rounds}"
        )

        for p, reference in (
            expected[rounds].items()
        ):

            value = exact_temporal_decoder(
                p,
                rounds
            )

            difference = abs(
                value - reference
            )

            print(
                f"p={p:.3f} | "
                f"new={value:.12f} | "
                f"old={reference:.12f} | "
                f"diff={difference:.3e}"
            )

            if not np.isclose(
                value,
                reference,
                atol=1e-10
            ):

                passed = False

    if passed:

        print(
            "\n✅ R=2 and R=3 reproduce Stage 50.9"
        )

    else:

        print(
            "\n❌ Cross-validation mismatch"
        )

    return passed


# ======================================================================
# 17. ROUND SCALING
# ======================================================================

def round_scaling():

    print("\n" + "=" * 70)
    print(
        "TEST 3: Exact Round Scaling"
    )
    print("=" * 70)

    noise_levels = [
        0.001,
        0.002,
        0.005,
        0.010,
        0.020,
        0.050,
    ]

    rounds_list = [
        1,
        2,
        3,
        4,
        5,
        10,
    ]

    results = []

    for rounds in rounds_list:

        print(
            f"\nRounds = {rounds}"
        )

        print(
            "p        | P_L              | P_L/p²"
        )

        print(
            "---------|-------------------|------------"
        )

        for p in noise_levels:

            (
                pl,
                state_count,
                total_probability,
            ) = exact_temporal_decoder(
                p,
                rounds,
                return_state_statistics=True
            )

            ratio = (
                pl / (p * p)
            )

            print(
                f"{p:.3f}    | "
                f"{pl:.12f} | "
                f"{ratio:.6f}"
            )

            results.append(
                {
                    "rounds": rounds,
                    "p": p,
                    "pl": pl,
                    "ratio": ratio,
                    "states": state_count,
                }
            )

        # Low-noise estimate from first two points.
        low_noise = [
            x for x in results
            if x["rounds"] == rounds
            and x["p"] <= 0.002
        ]

        if low_noise:

            coefficient = np.mean(
                [
                    x["ratio"]
                    for x in low_noise
                ]
            )

            print(
                f"\nEstimated low-noise coefficient: "
                f"{coefficient:.6f}"
            )

            print(
                f"Expected 10R: "
                f"{10 * rounds:.6f}"
            )

    return results


# ======================================================================
# 18. LINEAR SCALING TEST
# ======================================================================

def test_linear_round_scaling(
    results
):

    print("\n" + "=" * 70)
    print(
        "TEST 4: A_R vs R"
    )
    print("=" * 70)

    estimates = []

    for rounds in sorted(
        set(
            r["rounds"]
            for r in results
        )
    ):

        low_noise = [
            r["ratio"]
            for r in results
            if r["rounds"] == rounds
            and r["p"] <= 0.002
        ]

        if not low_noise:
            continue

        estimate = np.mean(
            low_noise
        )

        estimates.append(
            (
                rounds,
                estimate
            )
        )

        print(
            f"R={rounds:2d} | "
            f"A_R≈{estimate:.6f} | "
            f"10R={10*rounds:.6f} | "
            f"A_R/(10R)="
            f"{estimate/(10*rounds):.6f}"
        )

    if estimates:

        deviations = [
            abs(
                a / (10 * r) - 1
            )
            for r, a in estimates
        ]

        max_deviation = max(
            deviations
        )

        print(
            f"\nMaximum relative deviation "
            f"from 10R: "
            f"{max_deviation:.6e}"
        )


# ======================================================================
# 19. MAIN
# ======================================================================

def run_stage_51():

    print("=" * 70)
    print(
        "مرحله 51: Efficient Exact Temporal Decoder"
    )
    print(
        "[[5,1,3]]"
    )
    print("=" * 70)

    print(
        "\nMethod:"
    )

    print(
        "   1024 physical Pauli errors"
    )

    print(
        "   ↓"
    )

    print(
        "   16 syndrome-conditioned logical kernels"
    )

    print(
        "   ↓"
    )

    print(
        "   4-state logical belief"
    )

    print(
        "   ↓"
    )

    print(
        "   dynamic programming"
    )

    print(
        "   ↓"
    )

    print(
        "   exact P_L(p,R)"
    )

    # ----------------------------------------------------------
    # TEST 1
    # ----------------------------------------------------------

    if not test_r1():

        print(
            "\n❌ STAGE 51 FAILED at TEST 1"
        )

        return

    # ----------------------------------------------------------
    # TEST 2
    # ----------------------------------------------------------

    if not test_r2_r3():

        print(
            "\n❌ STAGE 51 FAILED at TEST 2"
        )

        return

    # ----------------------------------------------------------
    # TEST 3
    # ----------------------------------------------------------

    results = round_scaling()

    # ----------------------------------------------------------
    # TEST 4
    # ----------------------------------------------------------

    test_linear_round_scaling(
        results
    )

    # ----------------------------------------------------------
    # FINAL
    # ----------------------------------------------------------

    print("\n" + "=" * 70)
    print(
        "🏆 STAGE 51 COMPLETE"
    )
    print("=" * 70)

    print("""
✅ R=1 reproduces Stage 28 exactly
✅ R=2/R=3 reproduce Stage 50.9
✅ Explicit syndrome histories are no longer enumerated
✅ Exact temporal Bayesian decoder constructed
✅ Round scaling measured
✅ A_R vs R evaluated

Main scientific question:
    Does A_R = 10R?

This determines whether the leading-order logical-error
coefficient grows linearly with the number of syndrome rounds.
""")

    print("=" * 70)


if __name__ == "__main__":
    run_stage_51()