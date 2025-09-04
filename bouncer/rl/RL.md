Reinforcement Learning Spec

This document specifies the v1 RL interface and signals for training a bouncer policy on LocalGame.

Spaces

- Action: `Discrete(2)` with `0=deny`, `1=accept`.
- Observation: `Box(low=0, high=1, shape=(31,), dtype=float32)`.

Attribute Order (fixed union of all scenarios)

1. `young`
2. `well_dressed`
3. `techno_lover`
4. `well_connected`
5. `creative`
6. `berlin_local`
7. `underground_veteran`
8. `international`
9. `fashion_forward`
10. `queer_friendly`
11. `vinyl_collector`
12. `german_speaker`

Attributes are disjoint across scenarios and this union will not change.

Observation Layout (31 floats)

- Indices 0–2: Scenario one-hot `[s1, s2, s3]` (e.g., scenario 2 → `[0,1,0]`).
- Index 3: Admitted fraction `admitted / CAPACITY`.
- Index 4: Remaining slots `(CAPACITY - admitted) / CAPACITY`.
- Index 5: Rejection pressure `rejected / REJECTION_LIMIT`.
- Index 6: Feasibility slack `max(0, remaining_slots_abs - sum(deficits_abs)) / CAPACITY` where `deficits_abs = sum(max(0, minCount_a - accepted_count_a))`.
- Indices 7–30: Per-attribute pairs in the fixed order above. For attribute `i` (0-based in the list), indices `7 + 2*i` and `8 + 2*i`:
  - `curr_bit_i`: 1 if current person has the attribute, else 0.
  - `remain_need_i`: `max(0, minCount_i - accepted_count_i) / CAPACITY` (0 if not constrained in the current scenario).

Notes

- Counts are computed over accepted persons only (`decision=True`).
- Attributes not present in the current scenario naturally yield zeros for both features.
- Correlations and relative frequencies are not included in v1; scenario one-hot provides the necessary context.

Rewards

- Per-step:
  - Reject (action `0`): `-1`.
  - Accept (action `1`): `0`.
- Terminal:
  - Success (capacity filled, all minima met): `0` additional.
  - Failure due to unmet minima at capacity: `-1000` additional on the final transition.
  - Failure due to rejection limit: `0` additional (per-reject penalties already accrued).

Termination

An episode ends when either:

- Capacity is reached. If all minima are met → success; otherwise → failure.
- Rejection limit is reached → failure.

Gymnasium API contracts:

- Return `(terminated=True, truncated=False)` on all end conditions.

Wrapper Notes

- The environment wraps `LocalGame` from `bouncer.models` and expects to run in a Django context (e.g., via `manage.py` or with `django.setup()` completed).
- To reduce database load, the wrapper maintains its own in-memory counters for total admitted/rejected and per-attribute accepted counts, updating them on each action.

