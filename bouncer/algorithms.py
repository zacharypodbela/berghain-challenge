"""
Berghain Challenge Algorithms

This module contains different algorithms for making accept/reject decisions
on people trying to get into the nightclub.
"""

import math
from collections.abc import Callable
from typing import TextIO

from django.core.management.base import CommandError
from stable_baselines3 import PPO

from bouncer.constants import CAPACITY
from bouncer.models import Game, Person
from bouncer.rl.env import ATTRIBUTE_ORDER, build_observation_vector

AlgorithmFunc = Callable[[Person, Game, TextIO, str | None], bool]


def too_nice_bouncer(
    person: Person, game: Game, stdout: TextIO, model_path: str | None
) -> bool:
    """
    Simple algorithm that always accepts everyone.

    Args:
        person: Person model instance with attributes
        game: Game model instance with constraints and statistics

    Returns:
        bool: True to accept, False to reject
    """
    return True


def so_mean_bouncer(
    person: Person, game: Game, stdout: TextIO, model_path: str | None
) -> bool:
    """
    Simple algorithm that always rejects everyone.

    Args:
        person: Person model instance with attributes
        game: Game model instance with constraints and statistics

    Returns:
        bool: True to accept, False to reject
    """
    return False


def optimal_markov_bouncer(
    person: Person, game: Game, stdout: TextIO, model_path: str | None
) -> bool:
    """
    Markov Decision Process approach to minimize rejections while meeting constraints.

    Strategy:
    1. Always accept people who satisfy both constraints (most efficient)
    2. Use dynamic thresholds based on current state and remaining capacity
    3. Balance constraint fulfillment to avoid getting stuck
    4. Increase acceptance rate as rejection limit approaches

    Args:
        person: Person model instance with attributes
        game: Game model instance with constraints and statistics

    Returns:
        bool: True to accept, False to reject
    """
    # Parse constraints - assuming scenario 1 structure
    constraints = {c["attribute"]: c["minCount"] for c in game.constraints}

    # Current state from game
    admitted = game.admitted_count
    rejected = game.rejected_count
    pending = game.pending_count
    total_seen = admitted + rejected + pending

    # Constraint tracking (assuming young/well_dressed for scenario 1)
    young_needed = max(
        0,
        constraints.get("young", 0)
        - sum(
            1
            for p in game.people.filter(decision=True)
            if p.attributes.get("young", False)
        ),
    )

    well_dressed_needed = max(
        0,
        constraints.get("well_dressed", 0)
        - sum(
            1
            for p in game.people.filter(decision=True)
            if p.attributes.get("well_dressed", False)
        ),
    )

    # Remaining capacity
    capacity_remaining = 1000 - admitted

    # Person attributes
    is_young = person.attributes.get("young", False)
    is_well_dressed = person.attributes.get("well_dressed", False)

    # Type classification
    has_both = is_young and is_well_dressed
    has_neither = not is_young and not is_well_dressed

    # ALWAYS accept people with both attributes (most efficient)
    if has_both:
        return True

    # NEVER accept people with neither if we still need both constraints
    if has_neither and young_needed > 0 and well_dressed_needed > 0:
        return False

    # If no capacity left, reject
    if capacity_remaining <= 0:
        return False

    # If close to rejection limit, be more accepting
    rejection_pressure = rejected / 20000
    urgency_multiplier = 1.0 + (rejection_pressure**2) * 3.0

    # Calculate constraint urgency
    young_urgency = young_needed / max(1, capacity_remaining) if young_needed > 0 else 0
    well_dressed_urgency = (
        well_dressed_needed / max(1, capacity_remaining)
        if well_dressed_needed > 0
        else 0
    )

    # Dynamic threshold based on game progress
    game_progress = admitted / 1000
    base_threshold = 0.3 - (game_progress * 0.2)  # Start selective, get more accepting

    # Decision logic for single-attribute people
    if is_young and not is_well_dressed:
        if young_needed == 0:
            # We've met young quota, only accept if we have lots of capacity
            return bool(capacity_remaining > (well_dressed_needed + 100))

        # Accept based on urgency and game state
        threshold = base_threshold / (urgency_multiplier * (1 + young_urgency))
        acceptance_rate = admitted / max(1, total_seen)
        return acceptance_rate < (1 - threshold)

    if is_well_dressed and not is_young:
        if well_dressed_needed == 0:
            # We've met well_dressed quota, only accept if we have lots of capacity
            return bool(capacity_remaining > (young_needed + 100))

        # Accept based on urgency and game state
        threshold = base_threshold / (urgency_multiplier * (1 + well_dressed_urgency))
        acceptance_rate = admitted / max(1, total_seen)
        return acceptance_rate < (1 - threshold)

    # People with neither attribute - only accept if we've met both constraints
    # and have reasonable capacity left
    if young_needed == 0 and well_dressed_needed == 0:
        return capacity_remaining > 50  # Keep some buffer

    return False


# --- Helper: classify incoming person into the four buckets ---
def _classify_person(
    person: Person, y_key: str = "young", w_key: str = "well_dressed"
) -> str:
    y = bool(person.attributes.get(y_key, False))
    w = bool(person.attributes.get(w_key, False))
    if y and w:
        return "both"
    if y and not w:
        return "young_only"
    if (not y) and w:
        return "well_only"
    return "neither"


# --- Helper: pull constraints with proper parsing ---
def _parse_constraints(game: Game) -> tuple[int, int, int, int]:
    # Parse constraints like other algorithms
    constraints = {c["attribute"]: c["minCount"] for c in game.constraints}
    total = 1000  # Fixed total capacity
    min_y = constraints.get("young", 600)  # Default fallback
    min_w = constraints.get("well_dressed", 600)  # Default fallback
    buffer_slots = 0  # No extra safety buffer by default
    return total, min_y, min_w, buffer_slots


# --- Helper: compute live counters from accepted people so far ---
def _compute_counters(
    game: Game, y_key: str = "young", w_key: str = "well_dressed"
) -> tuple[int, int, int, int, int]:
    accepted = game.people.filter(decision=True)
    A = accepted.count()

    B = Y1 = W1 = 0
    for p in accepted:
        y = bool(p.attributes.get(y_key, False))
        w = bool(p.attributes.get(w_key, False))
        if y and w:
            B += 1
        elif y and not w:
            Y1 += 1
        elif (not y) and w:
            W1 += 1
        else:
            # neither (we track implicitly via N below)
            pass
    N = A - (B + Y1 + W1)
    return A, B, Y1, W1, N


def two_trait_heuristic_bouncer(
    person: Person, game: Game, stdout: TextIO, model_path: str | None
) -> bool:
    """
    Online decision rule with 'both-credit' and 'debt' guards.

    Guarantees feasibility (>= min_young, >= min_well, == total) while minimizing rejections in expectation.
    - Always accept BOTH (if slots remain).
    - Pay down whichever debt (Y/W) is still positive with the corresponding one-only.
    - Admit NEITHER only if you have BOTH-credit and still enough slots to satisfy remaining debts.
    - Protect remaining slots so you never get cornered late.
    """
    y_key = "young"
    w_key = "well_dressed"

    # Parse constraints
    total, min_y, min_w, buffer_slots = _parse_constraints(game)

    # Early exit if game is already finished
    if game.status != "running":
        return False

    # Tally live state
    A, B, Y1, W1, N = _compute_counters(game, y_key, w_key)
    S = total - A  # remaining slots

    if S <= 0:
        # No capacity left
        return False

    category = _classify_person(person, y_key, w_key)

    # Compute debts and credits
    y_accepted = B + Y1
    w_accepted = B + W1
    DY = max(0, min_y - y_accepted)  # how many more Y (of any kind) we still owe
    DW = max(0, min_w - w_accepted)  # how many more W (of any kind) we still owe

    # Required overlap implied by inclusion-exclusion:
    # to hit both minima inside a fixed total, you need at least this many BOTH.
    required_overlap = max(0, min_y + min_w - total)
    both_credit = max(0, B - required_overlap)  # each NEITHER spends 1 credit

    # --- Decision logic ---
    accept = False

    if category == "both":
        # Best type: helps both constraints at once. Always accept if capacity remains.
        accept = True

    elif category == "young_only":
        if DY > 0:
            accept = True  # pay down Y debt first
        else:
            # No Y debt; only accept if you can still cover remaining W debt after this admit
            # (i.e., don't steal a slot that DW might need).
            accept = (S - 1) >= (DW + buffer_slots)

    elif category == "well_only":
        if DW > 0:
            accept = True  # pay down W debt first
        else:
            # No W debt; protect Y debt if any
            accept = (S - 1) >= (DY + buffer_slots)

    else:  # category == "neither"
        # Safe to accept only if:
        # 1) we have spare BOTH credit (each NEITHER consumes one),
        # 2) after taking this slot, we still have enough capacity to pay both debts.
        has_credit = N < both_credit
        protects_debts = (S - 1) >= (DY + DW + buffer_slots)
        accept = has_credit and protects_debts

    return accept


def deficit_weighted_bouncer(
    person: Person, game: Game, stdout: TextIO, model_path: str | None
) -> bool:
    """Dynamic-overlap heuristic tuned for multi-constraint scenarios.

    If total remaining deficits are D and remaining capacity is S, each admitted person
    must cover on average D/S needed attributes. Accept only if the current person covers
    at least ceil(D/S) needed attributes, with feasibility guards.
    """
    # Min counts
    constraints = {c["attribute"]: int(c["minCount"]) for c in game.constraints}
    min_counts = {a: int(constraints.get(a, 0)) for a in ATTRIBUTE_ORDER}

    # Accepted counts so far
    accepted_attr_counts: dict[str, int] = {}
    for attr in ATTRIBUTE_ORDER:
        accepted_attr_counts[attr] = game.people.filter(
            decision=True, **{f"attributes__{attr}": True}
        ).count()

    admitted = int(game.admitted_count)
    remaining_capacity = max(0, CAPACITY - admitted)

    # Compute deficits and overlap with needed attributes
    total_deficits = 0
    deficits: dict[str, int] = {}
    for attr in ATTRIBUTE_ORDER:
        need = max(0, min_counts.get(attr, 0) - accepted_attr_counts.get(attr, 0))
        deficits[attr] = need
        total_deficits += need

    # If all minima met, admit remaining people
    if total_deficits == 0:
        return True

    overlap_needed = 0
    for attr in ATTRIBUTE_ORDER:
        if deficits[attr] > 0 and bool(person.attributes.get(attr, False)):
            overlap_needed += 1

    # Reject if nothing helpful
    if overlap_needed == 0:
        return False

    # Required minimum overlap based on average coverage needed
    if remaining_capacity <= 0:
        return False
    avg_needed = float(total_deficits) / float(remaining_capacity)
    required_k = max(1, math.ceil(avg_needed))
    return bool(overlap_needed >= required_k)


def deficit_weighted_bouncer_s3(
    person: Person, game: Game, stdout: TextIO, model_path: str | None
) -> bool:
    """A variant of deficit_weighted_bouncer with special rules to win scenario 3."""
    constraints = {c["attribute"]: int(c["minCount"]) for c in game.constraints}
    min_counts = {a: int(constraints.get(a, 0)) for a in ATTRIBUTE_ORDER}

    # Accepted counts so far
    accepted_attr_counts: dict[str, int] = {}
    for attr in ATTRIBUTE_ORDER:
        accepted_attr_counts[attr] = game.people.filter(
            decision=True, **{f"attributes__{attr}": True}
        ).count()

    admitted = int(game.admitted_count)
    remaining_capacity = max(0, CAPACITY - admitted)

    # Deficits
    deficits: dict[str, int] = {}
    total_deficits = 0
    for attr in ATTRIBUTE_ORDER:
        need = max(0, min_counts.get(attr, 0) - accepted_attr_counts.get(attr, 0))
        deficits[attr] = need
        total_deficits += need

    if total_deficits == 0:
        return True

    needed_attributes = [attr for attr in ATTRIBUTE_ORDER if deficits[attr] > 0]
    needed_rare_attributes = [
        attr
        for attr in needed_attributes
        if attr in ["queer_friendly", "vinyl_collector", "international"]
    ]

    # These hardcoded rules run for 90% of the game, then the generic deficit-weighted heuristic takes over to finish
    if deficits["german_speaker"] > 0:
        num_needed_rare_attributes_in_curr_person = sum(
            1
            for attr in needed_rare_attributes
            if bool(person.attributes.get(attr, False))
        )
        num_needed_rare_attributes = sum(1 for attr in needed_rare_attributes)
        if bool(person.attributes.get("german_speaker", False)):
            return (
                num_needed_rare_attributes_in_curr_person
                >= num_needed_rare_attributes - 1
            )
        else:
            if num_needed_rare_attributes <= 1:
                return False
            else:
                return (
                    num_needed_rare_attributes_in_curr_person
                    == num_needed_rare_attributes
                )

    # Needed-overlap for this person
    needed_present = [
        attr
        for attr in ATTRIBUTE_ORDER
        if deficits[attr] > 0 and bool(person.attributes.get(attr, False))
    ]
    overlap = len(needed_present)
    if overlap == 0 or remaining_capacity <= 0:
        return False

    k = max(1, math.ceil(float(total_deficits) / float(remaining_capacity)))
    if overlap >= k + 1:
        return True

    # Urgent set: top-2 deficits
    urgent = [
        a
        for a, d in sorted(deficits.items(), key=lambda x: x[1], reverse=True)
        if d > 0
    ][:2]
    if overlap >= k and any(a in urgent for a in needed_present):
        return True

    # Coverage share criterion (focus on largest deficits)
    covered_deficit = sum(deficits[a] for a in needed_present)
    coverage_share = float(covered_deficit) / float(max(1, total_deficits))
    if overlap >= k and coverage_share >= 0.6:
        return True

    return False


def ppo_bouncer(
    person: Person, game: Game, stdout: TextIO, model_path: str | None
) -> bool:
    """Use a saved Stable-Baselines3 PPO policy to decide."""
    if model_path is None:
        raise CommandError("ppo_bouncer requires --model-path argument")
    model: PPO = PPO.load(model_path)

    # Build observation vector from Game state
    admitted = int(game.admitted_count)
    rejected = int(game.rejected_count)
    constraints = {c["attribute"]: int(c["minCount"]) for c in game.constraints}
    min_counts = {a: int(constraints.get(a, 0)) for a in ATTRIBUTE_ORDER}
    accepted_attr_counts: dict[str, int] = {}
    for attr in ATTRIBUTE_ORDER:
        accepted_attr_counts[attr] = game.people.filter(
            decision=True, **{f"attributes__{attr}": True}
        ).count()

    obs = build_observation_vector(
        scenario=int(game.scenario),
        admitted=admitted,
        rejected=rejected,
        min_counts=min_counts,
        accepted_attr_counts=accepted_attr_counts,
        current_attrs=person.attributes,
    )

    # Give the model the observation vector and ask for next action
    action, _ = model.predict(obs, deterministic=True)
    # Action space is Discrete(2): 1 = accept, 0 = reject
    return bool(int(action) == 1)


# Registry of available algorithms
ALGORITHMS: dict[str, AlgorithmFunc] = {
    "too_nice_bouncer": too_nice_bouncer,
    "so_mean_bouncer": so_mean_bouncer,
    "optimal_markov_bouncer": optimal_markov_bouncer,
    "two_trait_heuristic_bouncer": two_trait_heuristic_bouncer,
    "deficit_weighted_bouncer": deficit_weighted_bouncer,
    "deficit_weighted_bouncer_s3": deficit_weighted_bouncer_s3,
    "ppo_bouncer": ppo_bouncer,
}


def get_algorithm(name: str) -> AlgorithmFunc:
    """Get an algorithm by name"""
    if name not in ALGORITHMS:
        raise ValueError(
            f"Unknown algorithm: {name}. Available: {list(ALGORITHMS.keys())}"
        )
    return ALGORITHMS[name]
