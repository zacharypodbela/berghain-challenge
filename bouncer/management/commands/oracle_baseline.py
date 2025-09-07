from __future__ import annotations

from typing import Any, TextIO

import numpy as np
from django.core.management.base import BaseCommand
from ortools.sat.python import cp_model

from bouncer.constants import CAPACITY, REJECTION_LIMIT, SCENARIO_CONFIGS
from bouncer.generate_attributes import CorrelatedAttributeGenerator


def _solve_prefix_feasible(
    population: list[dict[str, bool]], constraints: dict[str, int], capacity: int
) -> bool:
    """
    Exact feasibility via MILP/CP-SAT: Is there a subset of exactly `capacity`
    people in the `population` that meets all minima in `constraints`? Returns True/False.
    """
    needed_attrs = list(constraints.keys())
    # For any attribute, if the prefix does not contain enough of that attribute to meet its minimum → infeasible
    for a in needed_attrs:
        have = sum(1 for p in population if bool(p.get(a, False)))
        if have < int(constraints[a]):
            return False

    # Split population into useful (has at least one needed attr) and fillers (no needed attrs)
    useful_indices: list[int] = []
    for i, p in enumerate(population):
        if any(bool(p.get(a, False)) for a in needed_attrs):
            useful_indices.append(i)

    # Build incidence matrix only for useful people
    M: list[list[int]] = [
        [1 if bool(population[i].get(a, False)) else 0 for a in needed_attrs]
        for i in useful_indices
    ]
    n_useful = len(useful_indices)

    # Quick constructive greedy: try to meet deficits using useful people only
    deficits = {a: int(constraints[a]) for a in needed_attrs}
    selected: list[int] = []
    remaining = set(range(n_useful))
    while (
        remaining
        and any(deficits[a] > 0 for a in needed_attrs)
        and len(selected) < capacity
    ):
        # pick argmax marginal coverage of remaining deficits
        best_i = None
        best_gain = -1
        for i in list(remaining):
            gain = 0
            row = M[i]
            for j, a in enumerate(needed_attrs):
                if deficits[a] > 0 and row[j] == 1:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_i = i
        if best_i is None or best_gain <= 0:
            break
        selected.append(best_i)
        remaining.remove(best_i)
        row = M[best_i]
        for j, a in enumerate(needed_attrs):
            if row[j] == 1 and deficits[a] > 0:
                deficits[a] -= 1
    if all(deficits[a] <= 0 for a in needed_attrs) and len(selected) <= capacity:
        # We can always fill the rest with any fillers or extra useful; exact-capacity selection exists
        return True

    model = cp_model.CpModel()
    # Variables only for useful people
    x = [model.NewBoolVar(f"x_{i}") for i in range(n_useful)]
    # We only need to restrict useful selections to be <= capacity; the remaining slots can be filled with fillers
    model.Add(sum(x) <= int(capacity))
    for j, a in enumerate(needed_attrs):
        model.Add(
            sum(x[i] * int(M[i][j]) for i in range(n_useful)) >= int(constraints[a])
        )
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 5.0
    res = solver.Solve(model)
    if res in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return True
    if res == cp_model.INFEASIBLE:
        return False
    # UNKNOWN or MODEL_INVALID or other abnormal termination → raise
    status_name = solver.StatusName(res) if hasattr(solver, "StatusName") else str(res)
    raise RuntimeError(f"CP-SAT could not determine feasibility (status={status_name})")


def _oracle_rejections_for_trial(
    generator: CorrelatedAttributeGenerator,
    constraints: dict[str, int],
    seed: int,
    stdout: TextIO,
) -> int:
    # Start with lower bound CAPACITY and upper bound max_people
    low = CAPACITY  # Low is unknown feasibility
    hi = REJECTION_LIMIT + CAPACITY  # Hi is always feasible

    # Generate enough people for the worst case (hi)
    people: list[dict[str, bool]] = generator.sample(hi, seed=seed)

    # Binary search for minimum feasible prefix
    while hi != low:
        mid = (low + hi) // 2
        try:
            res = _solve_prefix_feasible(people[:mid], constraints, CAPACITY)
        except RuntimeError as e:
            # Surface CP-SAT abnormal status instead of silently treating as infeasible
            raise RuntimeError(f"Solver failure at prefix t={mid}: {e}") from e
        stdout.write(f"At {mid} people: {'feasible' if res else 'infeasible'}\n")
        if res:
            hi = mid
        else:
            low = mid + 1

    return int(hi - CAPACITY)


class Command(BaseCommand):
    help = (
        "Oracle baseline: simulate lower-bound rejections with perfect foresight "
        "(earliest feasible prefix), repeated over many trials."
    )

    def add_arguments(self, parser: Any) -> None:
        parser.add_argument("--scenario", type=int, choices=[1, 2, 3], required=True)
        parser.add_argument("--episodes", type=int, default=200)
        parser.add_argument("--seed", type=int, default=123)

    def handle(self, *args: Any, **opts: Any) -> None:
        scenario = int(opts["scenario"])
        n = int(opts["episodes"]) or 1
        seed0 = int(opts["seed"]) or 0

        self.stdout.write(
            self.style.MIGRATE_HEADING(
                f"Oracle baseline | scenario={scenario} episodes={n} seed={seed0}"
            )
        )

        cfg = SCENARIO_CONFIGS[scenario]
        constraints = {c["attribute"]: int(c["minCount"]) for c in cfg["constraints"]}
        generator = CorrelatedAttributeGenerator(cfg["attribute_statistics"])

        rejs: list[int] = []
        for i in range(n):
            r = _oracle_rejections_for_trial(
                generator,
                constraints,
                seed=seed0 + i * 17,
                stdout=self.stdout,
            )
            rejs.append(int(r))
            if (i + 1) % max(1, n // 10) == 0 or i == n - 1:
                self.stdout.write(
                    f"  Progress: {i + 1}/{n} | last_rejections={r} | mean_so_far={np.mean(rejs):.2f}"
                )

        arr = np.asarray(rejs, dtype=np.float64)
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        p90 = float(np.percentile(arr, 90.0))
        p95 = float(np.percentile(arr, 95.0))
        p99 = float(np.percentile(arr, 99.0))

        self.stdout.write("\nOracle Summary (Lower Bound)")
        self.stdout.write("----------------------------")
        self.stdout.write(f"Scenario:     {scenario}")
        self.stdout.write(f"Episodes:     {n}")
        self.stdout.write(f"Mean rejects: {mean:.2f}  std={std:.2f}")
        self.stdout.write(f"p90:          {p90:.2f}  p95={p95:.2f}  p99={p99:.2f}")
