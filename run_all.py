"""
Graph-Induced QEC Codes — Full Reproducibility Pipeline

This script runs all 22 pipeline steps in order.
Expected total runtime: ~30 minutes on a modern laptop.

Usage:
    python run_all.py
"""

import subprocess
import sys
from pathlib import Path


STEPS = [
    # Stage 1: Graph enumeration and subgroup search
    "01_graph_search/step_58_optimized_graph_search.py",
    "01_graph_search/complete_lc_perm_check.py",

    # Stage 2: Code analysis
    "02_code_analysis/step_20_exact_syndrome_correct.py",
    "02_code_analysis/step_23_final_group_equality.py",
    "02_code_analysis/step_24_independent_verification.py",
    "02_code_analysis/step_26_exact_decoder_fixed.py",
    "02_code_analysis/step_28_exact_full_polynomial.py",

    # Stage 3: Temporal two-fault analysis
    "03_temporal_analysis/step_50_9_exact_temporal_decoder.py",
    "03_temporal_analysis/step_51_efficient_exact_temporal_decoder.py",
    "03_temporal_analysis/step_52_analytical_leading_coefficient.py",
    "03_temporal_analysis/step_54_general_proof_N2R.py",

    # Stage 4: Machine-learning pipeline
    "04_ml_pipeline/build_proper_ml_dataset.py",
    "04_ml_pipeline/graph_disjoint_topology_qec.py",
    "04_ml_pipeline/model_2_final_corrected.py",
    "04_ml_pipeline/model_3a_corrected.py",
    "04_ml_pipeline/model_3_final_corrected.py",

    # Stage 5: Independent validation
    "05_validation/qec_independent_verification.py",
    "05_validation/full_history_collision_audit.py",
    "05_validation/stage_7_6_cross_round_logical_audit.py",
    "05_validation/step_43_genuine_stim_syndrome.py",

    # Stage 6: Verification tests
    "tests/test_01_equivalence_check.py",
    "tests/test_02_model_2_R2.py",
    "tests/test_03_N2_temporal.py",
]


def main():
    root = Path(__file__).resolve().parent
    print("=" * 70)
    print("Graph-Induced QEC Codes — Full Reproducibility Pipeline")
    print("=" * 70)

    for step in STEPS:
        path = root / step
        print()
        print("=" * 70)
        print(f"▶  {step}")
        print("=" * 70)

        if not path.exists():
            print(f"⚠️  Missing: {step}")
            continue

        result = subprocess.run(
            [sys.executable, str(path)],
            check=False,
            cwd=path.parent,
        )

        if result.returncode != 0:
            print()
            print(f"❌ FAILED: {step}")
            sys.exit(1)

    print()
    print("=" * 70)
    print("🎉 All steps completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
