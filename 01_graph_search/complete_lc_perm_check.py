"""
======================================================================
COMPLETE LC + PERMUTATION EQUIVALENCE FOR ALL 132 CODES
======================================================================
"""

import numpy as np
import pandas as pd
from itertools import permutations, product
import time


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


def weight_distribution(generators):
    group = build_group(generators)
    dist = {}
    for p in group:
        w = pauli_weight(p)
        dist[w] = dist.get(w, 0) + 1
    return tuple(sorted(dist.items()))


def apply_perm(pauli, perm):
    return ''.join(pauli[perm[i]] for i in range(len(perm)))


def apply_lc(pauli, lc_combo):
    return ''.join(lc_combo[i][pauli[i]] for i in range(len(pauli)))


def main():
    print("=" * 70)
    print("COMPLETE LC + PERMUTATION EQUIVALENCE CHECK")
    print("=" * 70)

    df = pd.read_csv("ml_full_dataset.csv")
    d3 = df[df['distance'] == 3].copy()
    print(f"\n📂 Total d=3 codes: {len(d3)}")

    CANONICAL = ["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"]
    canonical_group = build_group(CANONICAL)
    canonical_dist = weight_distribution(CANONICAL)

    print(f"\n📊 Canonical code:")
    print(f"   Generators: {CANONICAL}")
    print(f"   Group size: {len(canonical_group)}")
    print(f"   Weight distribution: {canonical_dist}")

    print("\n" + "=" * 70)
    print("STEP 1: WEIGHT DISTRIBUTION ANALYSIS")
    print("=" * 70)

    dist_to_codes = {}
    for idx, row in d3.iterrows():
        gen_str = row['stabilizer_generators']
        if pd.isna(gen_str):
            continue
        generators = gen_str.split(';')
        if len(generators) != 4:
            continue
        dist = weight_distribution(generators)
        edges = row['edges']
        if dist not in dist_to_codes:
            dist_to_codes[dist] = []
        dist_to_codes[dist].append({
            'graph_id': row['graph_id'],
            'edges': edges,
            'generators': generators,
        })

    print(f"\n📊 Unique weight distributions found: {len(dist_to_codes)}")
    for dist, codes in dist_to_codes.items():
        edge_counts = {}
        for c in codes:
            edge_counts[c['edges']] = edge_counts.get(c['edges'], 0) + 1
        matches = "✅ YES" if dist == canonical_dist else "❌ NO"
        print(f"\n   Distribution {dist}:")
        print(f"      Total codes: {len(codes)}")
        print(f"      Edge counts: {edge_counts}")
        print(f"      Matches canonical: {matches}")

    print("\n" + "=" * 70)
    print("STEP 2: FULL LC + PERMUTATION CHECK")
    print("=" * 70)

    candidates = []
    for dist, codes in dist_to_codes.items():
        if dist == canonical_dist:
            candidates.extend(codes)

    print(f"\n📌 Candidates (same weight distribution): {len(candidates)}")
    print(f"   Others (different distribution): {len(d3) - len(candidates)}")

    matched = []
    unmatched = []
    t0 = time.time()

    all_perms = list(permutations(range(5)))
    all_lc = list(product(EFFECTIVE_LC, repeat=5))

    for i, code in enumerate(candidates):
        generators = code['generators']
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
            matched.append(code)
        else:
            unmatched.append(code)

        if (i + 1) % 10 == 0 or (i + 1) == len(candidates):
            elapsed = time.time() - t0
            print(f"   Progress: {i+1}/{len(candidates)}, "
                  f"matched={len(matched)}, "
                  f"unmatched={len(unmatched)}, "
                  f"elapsed={elapsed:.1f}s")

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print(f"\n📊 Summary:")
    print(f"   Total d=3 codes: {len(d3)}")
    print(f"   Same weight distribution: {len(candidates)}")
    print(f"   Different weight distribution: {len(d3) - len(candidates)}")
    print()
    print(f"   Among candidates:")
    print(f"      LC+Perm equivalent: {len(matched)}")
    print(f"      NOT equivalent: {len(unmatched)}")

    print(f"\n📊 Matched by edge count:")
    matched_edges = {}
    for c in matched:
        matched_edges[c['edges']] = matched_edges.get(c['edges'], 0) + 1
    for e, count in sorted(matched_edges.items()):
        print(f"      {e} edges: {count}")

    print(f"\n📊 Unmatched by edge count:")
    unmatched_edges = {}
    for c in unmatched:
        unmatched_edges[c['edges']] = unmatched_edges.get(c['edges'], 0) + 1
    for e, count in sorted(unmatched_edges.items()):
        print(f"      {e} edges: {count}")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    total_equivalent = len(matched)
    total_not_equivalent = len(d3) - total_equivalent

    print(f"""
    از مجموع {len(d3)} کد با پارامتر [[5,1,3]]:

    ✅ {total_equivalent} کد تحت LC + Permutation هم‌ارز canonical
    ❌ {total_not_equivalent} کد کدهای کوانتومی متفاوت
    """)

    print("=" * 70)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()