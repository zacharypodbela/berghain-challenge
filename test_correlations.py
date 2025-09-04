import math
from itertools import combinations
from typing import Any

import numpy as np

from bouncer.math import generate_correlated_attributes


def test_generate_attribute_statistics(
    attribute_statistics: dict[str, Any],
) -> dict[str, Any]:
    """
    Heavy statistical test for generate_correlated_attributes.
    - Draws a very large sample
    - Verifies marginals and pairwise φ correlations match the input within sampling error
    - Validates that the input correlations are feasible for the given marginals

    Raises AssertionError on failure.
    Returns a small summary dict on success (useful when running outside pytest).
    """
    # ---- 0) Config ----------------------------------------------------------
    N = 2_000_000  # "very large"; bump to taste for tighter checks
    MARGINAL_SIGMAS = 5.0  # # of standard errors allowed on p-hat
    CORR_SIGMAS = 5.0  # # of standard errors allowed on Fisher-z
    MARGINAL_TOL_FLOOR = 2e-3
    CORR_Z_TOL_FLOOR = 0.010  # extra cushion for discreteness/nonlinearity
    NEAR_DEGENERATE_VAR = 1e-6  # skip corr checks if p(1-p) below this

    # ---- 1) Extract targets -------------------------------------------------
    attributes: list[str] = sorted(attribute_statistics["relativeFrequencies"].keys())
    k = len(attributes)
    p = np.array(
        [float(attribute_statistics["relativeFrequencies"][a]) for a in attributes],
        dtype=np.float64,
    )

    R_target = np.eye(k, dtype=np.float64)
    for i, ai in enumerate(attributes):
        row = attribute_statistics["correlations"][ai]
        for j, aj in enumerate(attributes):
            if i == j:
                continue
            r = float(row[aj])
            if not (-1.0 <= r <= 1.0):
                raise AssertionError(
                    f"Invalid correlation for ({ai},{aj}): {r} not in [-1,1]."
                )
            R_target[i, j] = r

    # Symmetry check (input should be symmetric; allow tiny numeric noise)
    if not np.allclose(R_target, R_target.T, atol=1e-8, rtol=0):
        diff_pairs = np.where(~np.isclose(R_target, R_target.T, atol=1e-8, rtol=0))
        examples = list(
            zip(diff_pairs[0].tolist()[:3], diff_pairs[1].tolist()[:3], strict=False)
        )
        raise AssertionError(
            f"Attribute correlation matrix not symmetric; example mismatches at indices: {examples}"
        )

    # ---- 2) Feasibility checks on target φ given marginals ------------------
    def phi_bounds(p_i: float, p_j: float):
        # Fréchet bounds on P11
        p11_min = max(0.0, p_i + p_j - 1.0)
        p11_max = min(p_i, p_j)
        var_i = p_i * (1.0 - p_i)
        var_j = p_j * (1.0 - p_j)
        denom = math.sqrt(max(var_i, 0.0) * max(var_j, 0.0))
        # If either variance is ~0, any correlation is undefined; skip pair later.
        if denom == 0.0:
            return -1.0, 1.0
        phi_min = (p11_min - p_i * p_j) / denom
        phi_max = (p11_max - p_i * p_j) / denom
        # Ensure ordering
        return (min(phi_min, phi_max), max(phi_min, phi_max))

    infeasible_msgs = []
    for i, j in combinations(range(k), 2):
        lo, hi = phi_bounds(p[i], p[j])
        r_tgt = R_target[i, j]
        if r_tgt < lo - 1e-6 or r_tgt > hi + 1e-6:
            infeasible_msgs.append(
                f"{attributes[i]}–{attributes[j]}: target φ={r_tgt:.4f} not in feasible [{lo:.4f}, {hi:.4f}] "
                f"for p=({p[i]:.4f},{p[j]:.4f})"
            )
    if infeasible_msgs:
        msg = "\n".join(infeasible_msgs[:8])
        raise AssertionError(
            "Input attribute_statistics has infeasible correlations:\n" + msg
        )

    # ---- 3) Generate a large sample ----------------------------------------
    people = generate_correlated_attributes(attribute_statistics, num_people=N)
    X = np.array(
        [[1 if person[a] else 0 for a in attributes] for person in people],
        dtype=np.float64,
    )
    # Sanity: shape
    if X.shape != (N, k):
        raise AssertionError(
            f"Generated shape mismatch: expected {(N, k)}, got {X.shape}"
        )

    # ---- 4) Check marginals -------------------------------------------------
    p_hat = X.mean(axis=0)
    se_p = np.sqrt(np.clip(p * (1.0 - p) / max(N, 1), 0.0, None))
    tol_p = np.maximum(MARGINAL_SIGMAS * se_p, MARGINAL_TOL_FLOOR)

    marginal_errs = {}
    for i, a in enumerate(attributes):
        diff = abs(p_hat[i] - p[i])
        marginal_errs[a] = float(diff)
        if diff > tol_p[i]:
            raise AssertionError(
                f"Marginal mismatch for {a}: p_hat={p_hat[i]:.5f}, target={p[i]:.5f}, "
                f"abs err={diff:.5f}, tol≈{tol_p[i]:.5f} (N={N})"
            )

    # ---- 5) Check pairwise correlations (φ) --------------------------------
    # Standardize columns to compute corr quickly
    var = np.clip(p_hat * (1.0 - p_hat), 0.0, None)
    std = np.sqrt(var)
    # Avoid divide-by-zero for degenerate columns
    safe_std = np.where(std < NEAR_DEGENERATE_VAR, 1.0, std)
    Z = (X - p_hat) / safe_std  # degenerates will be zeros and skipped

    R_hat = (Z.T @ Z) / (N - 1)
    np.fill_diagonal(R_hat, 1.0)

    def fisher_z(r: float) -> float:
        r = float(np.clip(r, -0.999999, 0.999999))
        return 0.5 * math.log((1.0 + r) / (1.0 - r))

    se_z = 1.0 / math.sqrt(max(N - 3, 1))
    tol_z = CORR_SIGMAS * se_z + CORR_Z_TOL_FLOOR

    corr_errs = {}
    for i, j in combinations(range(k), 2):
        # Skip if either variable is near-degenerate—correlation isn’t well-defined
        if std[i] < NEAR_DEGENERATE_VAR or std[j] < NEAR_DEGENERATE_VAR:
            continue
        r_hat = float(np.clip(R_hat[i, j], -1.0, 1.0))
        r_tgt = float(np.clip(R_target[i, j], -1.0, 1.0))
        dz = abs(fisher_z(r_hat) - fisher_z(r_tgt))
        corr_errs[f"{attributes[i]}::{attributes[j]}"] = float(abs(r_hat - r_tgt))
        if dz > tol_z:
            raise AssertionError(
                f"Correlation mismatch for {attributes[i]}–{attributes[j]}: "
                f"r_hat={r_hat:.4f}, target={r_tgt:.4f}, |Δz|={dz:.4f} > tol≈{tol_z:.4f} (N={N})"
            )

    return {
        "N": N,
        "attributes": attributes,
        "p_target": {a: float(p_i) for a, p_i in zip(attributes, p, strict=False)},
        "p_hat": {a: float(ph) for a, ph in zip(attributes, p_hat, strict=False)},
        "marginal_abs_err": marginal_errs,
        "max_marginal_abs_err": float(
            max(marginal_errs.values()) if marginal_errs else 0.0
        ),
        "pair_corr_abs_err": corr_errs,
        "max_pair_corr_abs_err": float(max(corr_errs.values()) if corr_errs else 0.0),
    }


if __name__ == "__main__":
    import bouncer.constants as consts

    for scenario_id, scenario in consts.SCENARIO_CONFIGS.items():
        print(f"Testing scenario {scenario_id}...")
        stats = test_generate_attribute_statistics(scenario["attribute_statistics"])
        print(f"  Passed for scenario {scenario_id}:")
        print(f"    Attributes: {stats['attributes']}")
        print(f"    Max marginal abs err: {stats['max_marginal_abs_err']:.5f}")
        print(f"    Max pair corr abs err: {stats['max_pair_corr_abs_err']:.5f}")
        print()
    print("All scenario tests passed.")
