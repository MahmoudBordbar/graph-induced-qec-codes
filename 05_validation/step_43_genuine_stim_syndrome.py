"""
مرحله 43: Genuine Stim Syndrome Validation
[[5,1,3]] Steane-style stabilizer-state preparation + MPP

هدف:
    برای تمام 4^5 = 1024 خطای Pauli،
    syndrome محاسبه‌شده توسط Python را با syndrome
    استخراج‌شده مستقیماً از Stim مقایسه کند.

این نسخه:
    1. یک code state معتبر برای [[5,1,3]] می‌سازد.
    2. خطای Pauli را روی state اعمال می‌کند.
    3. چهار stabilizer را با MPP اندازه می‌گیرد.
    4. نتیجه Stim را با syndrome دقیق Python مقایسه می‌کند.

هیچ syndromeای hard-code نشده است.
"""

import stim
import numpy as np
from itertools import product
from tqdm import tqdm


# ============================================================
# 1. [[5,1,3]] CODE
# ============================================================

STABILIZERS = [
    "XZZXI",
    "IXZZX",
    "XIXZZ",
    "ZXIXZ",
]

LOGICAL_X = "IIXYX"
LOGICAL_Z = "IIZXZ"


# ============================================================
# 2. Python EXACT SYNDROME
# ============================================================

def pauli_to_symplectic(pauli):
    n = len(pauli)

    z = np.zeros(n, dtype=np.uint8)
    x = np.zeros(n, dtype=np.uint8)

    for i, p in enumerate(pauli):
        if p == "X":
            x[i] = 1

        elif p == "Y":
            x[i] = 1
            z[i] = 1

        elif p == "Z":
            z[i] = 1

    return np.concatenate([z, x])


def symplectic_inner_product(v1, v2):

    n = len(v1) // 2

    z1 = v1[:n]
    x1 = v1[n:]

    z2 = v2[:n]
    x2 = v2[n:]

    return int(
        (np.dot(z1, x2) + np.dot(x1, z2)) % 2
    )


def compute_syndrome_python(error):

    syndrome = []

    v_error = pauli_to_symplectic(error)

    for stabilizer in STABILIZERS:

        v_stab = pauli_to_symplectic(stabilizer)

        syndrome.append(
            symplectic_inner_product(
                v_error,
                v_stab
            )
        )

    return tuple(syndrome)


# ============================================================
# 3. APPLY PAULI ERROR
# ============================================================

def append_pauli_error(circuit, error):

    for q, p in enumerate(error):

        if p == "X":
            circuit.append("X", [q])

        elif p == "Y":
            circuit.append("Y", [q])

        elif p == "Z":
            circuit.append("Z", [q])


# ============================================================
# 4. BUILD VALID [[5,1,3]] CODE STATE
# ============================================================

def build_code_state_tableau():

    """
    ساخت یک stabilizer state کامل.

    چهار stabilizer کد:
        S0
        S1
        S2
        S3

    به همراه logical Z:

        Z_L

    بنابراین 5 stabilizer مستقل برای 5 qubit داریم.

    این state یک |0_L> معتبر است.
    """

    stabilizers = [
        stim.PauliString("+XZZXI"),
        stim.PauliString("+IXZZX"),
        stim.PauliString("+XIXZZ"),
        stim.PauliString("+ZXIXZ"),

        # logical Z = +1
        stim.PauliString("+IIZXZ"),
    ]

    tableau = stim.Tableau.from_stabilizers(
        stabilizers
    )

    return tableau


# ============================================================
# 5. BUILD GENUINE STIM CIRCUIT
# ============================================================

def create_stim_circuit(error):

    """
    Circuit واقعی Stim:

        |0_L>
          ↓
        Pauli error
          ↓
        MPP(S0)
        MPP(S1)
        MPP(S2)
        MPP(S3)

    Measurement result:
        0 -> +1 eigenvalue
        1 -> -1 eigenvalue
    """

    # --------------------------------------------------------
    # ساخت state |0_L>
    # --------------------------------------------------------

    tableau = build_code_state_tableau()

    # Stim circuit برای preparation
    circuit = tableau.to_circuit()

    # --------------------------------------------------------
    # اعمال خطا
    # --------------------------------------------------------

    append_pauli_error(
        circuit,
        error
    )

    # --------------------------------------------------------
    # اندازه‌گیری مستقیم stabilizerها
    # --------------------------------------------------------

    for stabilizer in STABILIZERS:

        targets = []

        for q, p in enumerate(stabilizer):

            if p == "X":
                targets.append(
                    stim.target_x(q)
                )

            elif p == "Y":
                targets.append(
                    stim.target_y(q)
                )

            elif p == "Z":
                targets.append(
                    stim.target_z(q)
                )

            # I اصلاً وارد product نمی‌شود

            if p != "I":
                targets.append(
                    stim.target_combiner()
                )

        # آخرین combiner را حذف می‌کنیم
        if targets:
            targets.pop()

        circuit.append(
            "MPP",
            targets
        )

    return circuit


# ============================================================
# 6. STIM SYNDROME
# ============================================================

def compute_syndrome_stim(error):

    circuit = create_stim_circuit(error)

    sampler = circuit.compile_sampler()

    sample = sampler.sample(
        shots=1
    )

    # چهار MPP measurement آخر
    measurements = sample[0]

    syndrome = tuple(
        int(v)
        for v in measurements[-4:]
    )

    return syndrome


# ============================================================
# 7. CHECK CODE STATE
# ============================================================

def test_code_state():

    print("\n" + "=" * 70)
    print("TEST 1: Code-state verification")
    print("=" * 70)

    tableau = build_code_state_tableau()

    print("\nStabilizers of prepared state:")

    for s in tableau.to_stabilizers():

        print("   ", s)

    print("\nExpected:")

    print("    +XZZXI")
    print("    +IXZZX")
    print("    +XIXZZ")
    print("    +ZXIXZ")
    print("    +IIZXZ")

    print("\n✅ Code state constructed by Stim.")


# ============================================================
# 8. TEST NO-ERROR CASE
# ============================================================

def test_identity():

    print("\n" + "=" * 70)
    print("TEST 2: Identity error")
    print("=" * 70)

    error = "IIIII"

    py = compute_syndrome_python(error)

    stim_syndrome = compute_syndrome_stim(error)

    print("Error:", error)
    print("Python syndrome:", py)
    print("Stim syndrome:  ", stim_syndrome)

    if py == stim_syndrome:

        print("\n✅ Identity test PASSED")

        return True

    print("\n❌ Identity test FAILED")

    return False


# ============================================================
# 9. TEST ALL 1024 PAULI ERRORS
# ============================================================

def test_all_paulis():

    print("\n" + "=" * 70)
    print("TEST 3: Complete syndrome validation")
    print("=" * 70)

    all_paulis = [
        "".join(p)
        for p in product(
            ["I", "X", "Y", "Z"],
            repeat=5
        )
    ]

    print(
        f"\nTotal Pauli errors: {len(all_paulis)}"
    )

    passed = 0
    failed = 0

    failures = []

    for error in tqdm(
        all_paulis,
        desc="Checking"
    ):

        python_syndrome = compute_syndrome_python(
            error
        )

        stim_syndrome = compute_syndrome_stim(
            error
        )

        if python_syndrome == stim_syndrome:

            passed += 1

        else:

            failed += 1

            if len(failures) < 20:

                failures.append(
                    (
                        error,
                        python_syndrome,
                        stim_syndrome
                    )
                )

    print("\n" + "-" * 70)

    print(
        f"PASS: {passed}"
    )

    print(
        f"FAIL: {failed}"
    )

    if failures:

        print("\nFirst failures:")

        for error, py, st in failures:

            print(
                f"   {error}"
            )

            print(
                f"      Python = {py}"
            )

            print(
                f"      Stim   = {st}"
            )

    if failed == 0:

        print("\n" + "=" * 70)
        print("🎉 COMPLETE SUCCESS")
        print("=" * 70)

        print(
            "\n✅ 1024 / 1024 Pauli errors validated."
        )

        print(
            "✅ Python syndrome = Stim syndrome."
        )

        print(
            "✅ Genuine Stim syndrome extraction verified."
        )

        return True

    else:

        print("\n" + "=" * 70)
        print("❌ VALIDATION FAILED")
        print("=" * 70)

        return False


# ============================================================
# 10. MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "مرحله 43: Genuine Stim Syndrome Validation"
    )

    print(
        "[[5,1,3]] + Code State + MPP"
    )

    print("=" * 70)

    print(
        "\nStim version:",
        stim.__version__
    )

    # Test 1
    test_code_state()

    # Test 2
    identity_ok = test_identity()

    if not identity_ok:

        print(
            "\n❌ Identity test failed."
        )

        print(
            "Validation متوقف شد."
        )

        return

    # Test 3
    success = test_all_paulis()

    print("\n" + "=" * 70)

    if success:

        print(
            "🏆 مرحله 43 با موفقیت کامل شد."
        )

        print(
            "🏆 Stim syndrome extraction معتبر است."
        )

        print(
            "➡️ مرحله بعد: Exact Decoder + P_L"
        )

    else:

        print(
            "⚠️ مرحله 43 هنوز نیاز به بررسی دارد."
        )

    print("=" * 70)


if __name__ == "__main__":
    main()