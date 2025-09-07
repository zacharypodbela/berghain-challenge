from __future__ import annotations

import uuid
from typing import Any, TextIO

import numpy as np
from django.core.management.base import BaseCommand
from django.utils import timezone
from ortools.sat.python import cp_model

from bouncer.constants import CAPACITY, REJECTION_LIMIT, SCENARIO_CONFIGS
from bouncer.generate_attributes import CorrelatedAttributeGenerator
from bouncer.models import LocalGame, Person


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


def _find_feasible_selection(
    population: list[dict[str, bool]],
    constraints: dict[str, int],
    capacity: int,
) -> set[int] | None:
    """Find one feasible selection of exactly `capacity` indices meeting minima.

    Returns a set of global indices into the population (0..len(pop)-1) if feasible,
    otherwise None. Uses the same reductions as feasibility: first a greedy
    constructive attempt on useful people, then CP-SAT to extract a certificate,
    then fills up to exact capacity with earliest remaining people.
    """
    needed_attrs = list(constraints.keys())
    # Early necessary check
    for a in needed_attrs:
        have = sum(1 for p in population if bool(p.get(a, False)))
        if have < int(constraints[a]):
            return None

    useful_indices: list[int] = []
    filler_indices: list[int] = []
    for i, p in enumerate(population):
        if any(bool(p.get(a, False)) for a in needed_attrs):
            useful_indices.append(i)
        else:
            filler_indices.append(i)

    M = [
        [1 if bool(population[i].get(a, False)) else 0 for a in needed_attrs]
        for i in useful_indices
    ]

    # Greedy attempt
    deficits = {a: int(constraints[a]) for a in needed_attrs}
    sel_useful: list[int] = []
    remaining = set(range(len(useful_indices)))
    while (
        remaining
        and any(deficits[a] > 0 for a in needed_attrs)
        and len(sel_useful) < capacity
    ):
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
        sel_useful.append(best_i)
        remaining.remove(best_i)
        for j, a in enumerate(needed_attrs):
            if M[best_i][j] == 1 and deficits[a] > 0:
                deficits[a] -= 1
    if all(deficits[a] <= 0 for a in needed_attrs) and len(sel_useful) <= capacity:
        sel_global: list[int] = [useful_indices[i] for i in sel_useful]
        # Fill to capacity with earliest remaining (useful not selected first, then fillers)
        remaining_globals = [
            useful_indices[i] for i in sorted(remaining)
        ] + filler_indices
        for gi in remaining_globals:
            if len(sel_global) >= capacity:
                break
            if gi not in sel_global:
                sel_global.append(gi)
        return set(sel_global) if len(sel_global) == capacity else None

    # CP-SAT certificate on useful subset
    try:
        from ortools.sat.python import cp_model

        model = cp_model.CpModel()
        x = [model.NewBoolVar(f"x_{i}") for i in range(len(useful_indices))]
        model.Add(sum(x) <= int(capacity))
        for j, a in enumerate(needed_attrs):
            model.Add(
                sum(x[i] * int(M[i][j]) for i in range(len(useful_indices)))
                >= int(constraints[a])
            )
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 5.0
        res = solver.Solve(model)
        if res not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return None if res == cp_model.INFEASIBLE else None
        chosen_useful = [
            i for i in range(len(useful_indices)) if solver.Value(x[i]) == 1
        ]
        sel_global = [useful_indices[i] for i in chosen_useful]
        # Fill to capacity
        used = set(sel_global)
        for gi in useful_indices + filler_indices:
            if len(sel_global) >= capacity:
                break
            if gi not in used:
                sel_global.append(gi)
                used.add(gi)
        return set(sel_global) if len(sel_global) == capacity else None
    except Exception:
        return None


def _oracle_rejections_for_trial(
    generator: CorrelatedAttributeGenerator,
    constraints: dict[str, int],
    seed: int,
    stdout: TextIO,
) -> tuple[int, list[dict[str, bool]]]:
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

    return int(hi - CAPACITY), people[:hi]


class Command(BaseCommand):
    help = (
        "Oracle baseline: simulate lower-bound rejections with perfect foresight "
        "(earliest feasible prefix), repeated over many trials."
    )

    def add_arguments(self, parser: Any) -> None:
        parser.add_argument("--scenario", type=int, choices=[1, 2, 3], required=True)
        parser.add_argument("--episodes", type=int, default=200)
        parser.add_argument("--seed", type=int, default=123)
        parser.add_argument(
            "--csv-out",
            type=str,
            default="",
            help="Optional CSV path (seed,best_score). Defaults to oracle_s<scenario>_n<episodes>_seed<seed>.csv",
        )
        parser.add_argument(
            "--write-games",
            action="store_true",
            help="If set, writes perfectly played games to the DB for each successful episode. Tags: 'oracle-perfect' and 'seed:<seed>' will be added.",
        )

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
        failures = 0
        seeds: list[int] = []
        for i in range(n):
            try:
                r, prefix_people = _oracle_rejections_for_trial(
                    generator,
                    constraints,
                    seed=seed0 + i * 17,
                    stdout=self.stdout,
                )
            except RuntimeError as e:
                failures += 1
                self.stdout.write(
                    self.style.ERROR(
                        f"Episode {i + 1}/{n} failed (excluded): {type(e).__name__}: {e}"
                    )
                )
                continue
            rejs.append(int(r))
            seeds.append(int(seed0 + i * 17))
            # Periodic progress based on successful episodes only
            if (i + 1) % max(1, n // 10) == 0 or i == n - 1:
                mean_so_far = float(np.mean(rejs)) if rejs else float("nan")
                self.stdout.write(
                    f"  Progress: {i + 1}/{n} | successes={len(rejs)} failures={failures} "
                    f"| last_rejections={r} | mean_so_far={mean_so_far:.2f}"
                )

            if bool(opts.get("write_games", False)):
                # Persist a LocalGame with decisions for prefix_people up to t*, accepting indices in sel
                sel = _find_feasible_selection(prefix_people, constraints, CAPACITY)
                if sel is None:
                    # This should never happen since we just proved feasibility
                    self.stdout.write(
                        self.style.ERROR(
                            f"Episode {i + 1}/{n} unexpected error: could not reconstruct feasible selection despite solver proving feasibility. "
                            f"Seed: {seed0 + i * 17}. Skipping DB write."
                        )
                    )
                    continue
                game = LocalGame.objects.create(
                    game_id=str(uuid.uuid4()),
                    scenario=scenario,
                    constraints=cfg["constraints"],
                    attribute_statistics=cfg["attribute_statistics"],
                    status="completed",
                    completion_reason=f"Oracle perfect | rejects={int(r)}",
                    completed_at=timezone.now(),
                    tags=["oracle-perfect", f"seed:{seed0 + i * 17}"],
                )
                persons: list[Person] = []
                for idx, attrs in enumerate(prefix_people):
                    persons.append(
                        Person(
                            game=game,
                            person_index=idx,
                            attributes=attrs,
                            decision=True if idx in sel else False,
                        )
                    )
                # Bulk create in batches for efficiency
                Person.objects.bulk_create(persons, batch_size=1000)

        # Write raw results to CSV
        csv_out = str(opts.get("csv_out") or "").strip()
        if not csv_out:
            csv_out = f"datasets/oracle_s{scenario}_n{n}_seed{seed0}.csv"
        try:
            import csv
            import os

            os.makedirs(os.path.dirname(csv_out) or ".", exist_ok=True)
            with open(csv_out, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["seed", "best_score"])
                for s, r in zip(seeds, rejs, strict=False):
                    writer.writerow([s, int(r)])
            self.stdout.write(self.style.SUCCESS(f"Wrote CSV -> {csv_out}"))
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(
                    f"Failed to write CSV to {csv_out}: {type(e).__name__}: {e}"
                )
            )

        # Summarize results and print to stdout
        arr = np.asarray(rejs, dtype=np.float64)
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        min_r = float(np.min(arr))
        max_r = float(np.max(arr))
        p01 = float(np.percentile(arr, 1.0))
        p05 = float(np.percentile(arr, 5.0))
        p10 = float(np.percentile(arr, 10.0))
        p90 = float(np.percentile(arr, 90.0))
        p95 = float(np.percentile(arr, 95.0))
        p99 = float(np.percentile(arr, 99.0))

        self.stdout.write("\nOracle Summary (Lower Bound)")
        self.stdout.write("----------------------------")
        self.stdout.write(f"Scenario:     {scenario}")
        self.stdout.write(f"Episodes:     {n}")
        self.stdout.write(f"Successes:    {len(rejs)}")
        self.stdout.write(f"Failures:     {failures}")
        self.stdout.write(
            f"Rejects: mean={mean:.2f}  std={std:.2f}  min={min_r:.1f}  max={max_r:.1f}"
        )
        self.stdout.write(
            f"p01={p01:.2f}  p05={p05:.2f}  p10={p10:.2f}  |  p90={p90:.2f}  p95={p95:.2f}  p99={p99:.2f}"
        )
