# Graph-Induced Quantum Error-Correcting Codes

**Reproducibility package** for the manuscript:

> **Graph-Induced Quantum Error-Correcting Codes: Exhaustive Search, Temporal Decoding, and Leakage-Controlled Machine Learning**
>
> Mahmoud Bordbar  
> Central Laboratory and Student Research Center, Iran  
> September 2026

---

## Overview

This repository contains the complete source code and datasets required to
reproduce every numerical result reported in the manuscript:

- Exhaustive enumeration of all **728 connected labeled five-vertex graphs**
- Enumeration of all **31 rank-four stabilizer subgroups** per graph
- Exact code-distance computation for all **22,568 graph/subgroup constructions**
- Complete Pauli weight distribution for the **[[5,1,3]]** code
- Two-fault temporal analysis for **R = 1, 2, 3**
- Machine-learning pipeline with **graph-disjoint cross-validation**
- Independent algebraic verification of the code structure

---

## Installation

```bash
git clone https://github.com/MahmoudBordbar/graph-induced-qec-codes.git
cd graph-induced-qec-codes
pip install -r requirements.txt
```

**Tested environment:**
- Python 3.10 / 3.11
- NumPy 1.26.4
- NetworkX 3.2.1
- scikit-learn 1.4.2
- Stim 1.16.0
- PyMatching 2.4.0

---

## Quick Start

Run the full pipeline:

```bash
python run_all.py
```

Expected total runtime: **~30 minutes** on a modern laptop.

---

## Key Numerical Results

### Exhaustive Search (Section 3)

| Quantity | Value |
|---|---|
| Labeled five-vertex graphs | 1,024 |
| Connected labeled graphs | 728 |
| Rank-four subgroups per graph | 31 |
| Total constructions | 22,568 |
| Constructions with d ≥ 3 | 132 |
| Graph-isomorphism classes | 3 (12, 60, 60) |
| LC+Permutation equivalence | 132 / 132 |

### Complete Pauli Weight Distribution (Section 8)

| Weight | Total | Correctable | Logical Failure |
|---|---|---|---|
| 0 | 1 | 1 | 0 |
| 1 | 15 | 15 | 0 |
| 2 | 90 | 0 | 90 |
| 3 | 270 | 60 | 210 |
| 4 | 405 | 135 | 270 |
| 5 | 243 | 45 | 198 |
| **Total** | **1,024** | **256** | **768** |

### Temporal Scaling (Section 11)

| R | Same-round | Cross-round | Total | 90R² |
|---|---|---|---|---|
| 1 | 90 | 0 | 90 | 90 |
| 2 | 180 | 180 | 360 | 360 |
| 3 | 270 | 540 | 810 | 810 |

**Result:** N₂(D_full; R) = 90R² for R = 1, 2, 3.

### Machine-Learning Performance (Section 13)

| Feature Set | Features | Best Model | R² |
|---|---|---|---|
| Topology only | 6 | Random Forest | 0.104 ± 0.020 |
| Stabilizer only | 8 | Extra Trees | 0.338 ± 0.026 |
| **Topology + Stabilizer** | **14** | **Gradient Boosting** | **0.504 ± 0.022** |
| Combined + Engineered | 20 | Gradient Boosting | 0.503 ± 0.023 |

---

## Directory Structure

```
graph-induced-qec-codes/
├── README.md
├── LICENSE
├── requirements.txt
├── run_all.py
├── 01_graph_search/        (2 files)
├── 02_code_analysis/       (5 files)
├── 03_temporal_analysis/   (4 files)
├── 04_ml_pipeline/         (5 files)
├── 05_validation/          (4 files)
└── tests/                  (3 files)
```

---

## Verification Tests

The `tests/` directory contains three verification tests:

```bash
cd tests
python test_01_equivalence_check.py   # Expected: Matched 132, Unmatched 0
python test_02_model_2_R2.py          # Expected: R² = 0.3377, 0.3382, 0.3381
python test_03_N2_temporal.py         # Expected: N₂ = 90, 360, 810
```

All three should print `🎉 TEST PASSED`.

---

## License

MIT License — see [LICENSE](LICENSE).

## Citation

```bibtex
@article{Bordbar2026GraphQEC,
  author  = {Bordbar, Mahmoud},
  title   = {Graph-Induced Quantum Error-Correcting Codes:
             Exhaustive Search, Temporal Decoding, and
             Leakage-Controlled Machine Learning},
  year    = {2026},
  month   = {September},
  url     = {https://github.com/MahmoudBordbar/graph-induced-qec-codes}
}
```

## Contact

Mahmoud Bordbar — mahmoud.bordbar@gmail.com
