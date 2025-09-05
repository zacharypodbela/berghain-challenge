import math
from typing import Any

import numpy as np

import bouncer.constants as consts
from bouncer.generate_attributes import CorrelatedAttributeGenerator
from bouncer.models import Game


# ---- helpers used only in the test ----
def _fisher_z(r: float) -> float:
    r = float(np.clip(r, -0.999999, 0.999999))
    return 0.5 * math.log((1.0 + r) / (1.0 - r))


def _phi_bounds(p_i: float, p_j: float) -> tuple[float, float]:
    p11_min = max(0.0, p_i + p_j - 1.0)
    p11_max = min(p_i, p_j)
    denom = math.sqrt(max(p_i * (1.0 - p_i), 0.0) * max(p_j * (1.0 - p_j), 0.0))
    if denom == 0.0:
        return -1.0, 1.0
    phi_min = (p11_min - p_i * p_j) / denom
    phi_max = (p11_max - p_i * p_j) / denom
    return (min(phi_min, phi_max), max(phi_min, phi_max))


def test_generate_attribute_statistics(
    target_attribute_statistics: dict[str, Any],
    actual_population: list[dict[str, bool]],
) -> None:
    """
    • Assert marginals vs targets (binomial SE tolerance).
    • Assert pairwise φ vs targets (Fisher-z tolerance).
    """
    # ---- settings ----
    Z_MARG = 5.0
    Z_CORR = 5.0
    MIN_MARG_TOL = 1e-3
    EPS = 1e-9

    # ---- inputs ----
    N = len(actual_population)
    attrs: list[str] = sorted(target_attribute_statistics["relativeFrequencies"].keys())
    n = len(attrs)
    p = np.array(
        [target_attribute_statistics["relativeFrequencies"][a] for a in attrs],
        dtype=float,
    )

    # ---- format sample ----
    X = np.array(
        [[1 if person[a] else 0 for a in attrs] for person in actual_population],
        dtype=float,
    )

    # ---- marginals ----
    p_hat = X.mean(axis=0)
    se = np.sqrt(p * (1.0 - p) / N)
    marg_tol = np.maximum(MIN_MARG_TOL, Z_MARG * se)
    abs_err = np.abs(p_hat - p)

    bad_marg = [
        (attrs[i], float(p[i]), float(p_hat[i]), float(abs_err[i]), float(marg_tol[i]))
        for i in range(n)
        if abs_err[i] > marg_tol[i] + EPS
    ]
    if bad_marg:
        lines = [
            f"  {a}: p={pi:.4f}, p̂={ph:.4f}, |Δ|={de:.5f} > tol={to:.5f}"
            for (a, pi, ph, de, to) in bad_marg
        ]
        raise AssertionError("Marginal probability check failed:\n" + "\n".join(lines))

    # ---- correlations (strict vs targets) ----
    Phi_tgt = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            Phi_tgt[i, j] = Phi_tgt[j, i] = float(
                target_attribute_statistics["correlations"][attrs[i]][attrs[j]]
            )

    z_tol = Z_CORR / math.sqrt(max(N - 3.0, 1.0))
    violations = []
    worst_dz = 0.0

    for i in range(n):
        xi = X[:, i]
        for j in range(i + 1, n):
            xj = X[:, j]
            vi = xi.var()
            vj = xj.var()
            if vi <= EPS or vj <= EPS:
                continue

            r_hat = float(np.corrcoef(xi, xj)[0, 1])
            r_tgt = float(Phi_tgt[i, j])

            lo, hi = _phi_bounds(float(p[i]), float(p[j]))
            # We keep the test strict: if target violates Fréchet, report but still mark as violation.
            if r_tgt < lo - 1e-6 or r_tgt > hi + 1e-6:
                violations.append(
                    (attrs[i], attrs[j], r_hat, r_tgt, float("nan"), z_tol, lo, hi)
                )
                continue

            dz = abs(_fisher_z(r_hat) - _fisher_z(r_tgt))
            worst_dz = max(worst_dz, dz)
            if dz > z_tol + 1e-12:
                violations.append(
                    (attrs[i], attrs[j], r_hat, r_tgt, dz, z_tol, 0.0, 0.0)
                )

    if violations:
        who = ", ".join(
            [
                f"{a}–{b} (r̂={rh:+.4f}, r*={rt:+.4f}, |Δz|={dz if not math.isnan(dz) else float('nan'):.4f}>≈{zt:.4f})"
                for (a, b, rh, rt, dz, zt, lo, hi) in violations[:6]
            ]
        )
        extra = "" if len(violations) <= 6 else f" (+{len(violations) - 6} more)"
        raise AssertionError("Correlation check failed for pairs: " + who + extra)

    # ---- summary ----
    print("Passed:")
    print(f"  Attributes: {attrs}")
    print(f"  Max marginal abs err: {float(abs_err.max()):.6f}")
    print(f"  Max pair |Δz|: {worst_dz:.6f} (tol≈{z_tol:.6f})")


def test_all_scenarios(n: int) -> None:
    for scenario_id, scenario in consts.SCENARIO_CONFIGS.items():
        print(f"Testing scenario {scenario_id}...")
        attribute_statistics = scenario["attribute_statistics"]
        generator = CorrelatedAttributeGenerator(attribute_statistics)
        people = generator.sample(n)
        test_generate_attribute_statistics(attribute_statistics, people)
        print()


# Test our test by feeding it populations from server-side games and ensuring they pass
def get_population_attributes_from_game(game_id: str) -> list[dict[str, bool]]:
    game = Game.objects.get(game_id=game_id)
    return list(game.people.values_list("attributes", flat=True))


def validate_test_script_with_real_games() -> None:
    for game in Game.objects.all():
        print(f"Validating our test with real game data {game.game_id}...")
        scenario_config = consts.SCENARIO_CONFIGS[game.scenario]
        attribute_statistics = scenario_config["attribute_statistics"]
        test_generate_attribute_statistics(
            attribute_statistics,
            get_population_attributes_from_game(game.game_id),
        )
        print()
