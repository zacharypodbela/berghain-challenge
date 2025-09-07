from __future__ import annotations

from django.test import SimpleTestCase

from bouncer.management.commands.oracle_baseline import _solve_prefix_feasible


def _mk(
    prefix: list[str], mapping: dict[str, dict[str, bool]]
) -> list[dict[str, bool]]:
    return [mapping[k] for k in prefix]


class OracleFeasibilityTests(SimpleTestCase):
    def _assert_minimum_prefix(
        self,
        people: list[dict[str, bool]],
        cons: dict[str, int],
        cap: int,
        expected_t: int,
    ) -> None:
        for t in range(1, len(people) + 1):
            with self.subTest(t=t):
                assert _solve_prefix_feasible(people[:t], cons, cap) is bool(
                    t >= expected_t
                )

    def test_simple_ab_example_min_prefix(self) -> None:
        # capacity 5, population [A, B, None, None, AB, None], mins A=2, B=2
        # With EXACT capacity selection, feasible when prefix length >= 5
        mapping = {
            "A": {"A": True, "B": False},
            "B": {"A": False, "B": True},
            "AB": {"A": True, "B": True},
            "N": {"A": False, "B": False},
        }
        people = _mk(["A", "B", "N", "N", "AB", "N"], mapping)
        cons = {"A": 2, "B": 2}
        cap = 5
        expected_t = 5

        self._assert_minimum_prefix(people, cons, cap, expected_t=expected_t)

    def test_min_prefix_three_items(self) -> None:
        # capacity 3, people [A, B, AB], mins A=2, B=2 → t=3 feasible
        mapping = {
            "A": {"A": True, "B": False},
            "B": {"A": False, "B": True},
            "AB": {"A": True, "B": True},
        }
        people = _mk(["A", "B", "AB"], mapping)
        cons = {"A": 2, "B": 2}
        cap = 3
        expected_t = 3

        self._assert_minimum_prefix(people, cons, cap, expected_t=expected_t)

    def test_exact_minima_all_A(self) -> None:
        # capacity 3, want A=3, people [A, A, A] → feasible at t=3
        people = [{"A": True}, {"A": True}, {"A": True}]
        cons = {"A": 3}
        cap = 3
        expected_t = 3

        self._assert_minimum_prefix(people, cons, cap, expected_t=expected_t)

    def test_two_ab_satisfy(self) -> None:
        # capacity 2, mins A=2, B=1, people [AB, AB] → feasible at t=2
        people = [{"A": True, "B": True}, {"A": True, "B": True}]
        cons = {"A": 2, "B": 1}
        cap = 2
        expected_t = 2

        self._assert_minimum_prefix(people, cons, cap, expected_t=expected_t)

    def test_sum_minima_exceeds_capacity_disjoint(self) -> None:
        # capacity 3, mins A=2, B=2, C=2 with disjoint single-attribute people → infeasible
        mapping = {
            "A": {"A": True, "B": False, "C": False},
            "B": {"A": False, "B": True, "C": False},
            "C": {"A": False, "B": False, "C": True},
        }
        people = _mk(["A", "B", "C", "A", "B"], mapping)
        cons = {"A": 2, "B": 2, "C": 2}
        cap = 3
        expected_t = len(people) + 1  # Always infeasible

        self._assert_minimum_prefix(people, cons, cap, expected_t=expected_t)

    def test_two_As_over_three_people(self) -> None:
        # capacity 2, mins A=2, people [A, B, A] → feasible at t=3
        people = [{"A": True}, {"A": False}, {"A": True}]
        cons = {"A": 2}
        cap = 2
        expected_t = 3

        self._assert_minimum_prefix(people, cons, cap, expected_t=expected_t)
