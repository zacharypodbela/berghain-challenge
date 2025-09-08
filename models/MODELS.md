All evals run with same `--seed 12345`

# Scenario 1

**Best Mean (Always Win):** `ppo_bc_s1.zip`
**Best p90 (Risk On / Go For High Score):** `ppo_bc_s1_oracle500.zip`

## ppo_bc_s1.zip

Trained on 50 games played by `two_trait_heuristic_bouncer` (Dataset: `s1_expert_50.npz`).

Episodes: 1000

|          | Mean        | Std                 | P01               | P05      | P10      | P90     | P95     | P99     | Min       | Max     |
| -------- | ----------- | ------------------- | ----------------- | -------- | -------- | ------- | ------- | ------- | --------- | ------- |
| Reward   | -1058.26    | 1093.14             | -1108.00          | -1073.00 | -1059.10 | -939.00 | -925.00 | -892.95 | -20992.00 | -861.00 |
| length   | 1998.3      | 45.5                | 1893.0            | 1925.0   | 1939.0   | 2057.1  | 2072.1  | 2101.0  | 1861.0    | 2137.0  |
| Admitted | 1000.0      | 0.0                 | 1000.0            | 1000.0   | 1000.0   | 1000.0  | 1000.0  | 1000.0  | 1000.0    | 1000.0  |
| Rejected | 998.3       | 45.5                | 893.0             | 925.0    | 939.0    | 1057.1  | 1072.0  | 1101.0  | 861.0     | 1137.0  |
| Outcomes | success=997 | unmet_at_capacity=3 | rejection_limit=0 |

## ppo_s1_bc_v1.zip

**Summary:** No improvement over `ppo_bc_s1`.

Episodes: 1000

|          | Mean        | Std                 | P01               | P05      | P10      | P90     | P95     | P99     | Min       | Max     |
| -------- | ----------- | ------------------- | ----------------- | -------- | -------- | ------- | ------- | ------- | --------- | ------- |
| Reward   | -1058.26    | 1093.14             | -1108.00          | -1073.00 | -1059.10 | -939.00 | -925.00 | -892.95 | -20992.00 | -861.00 |
| length   | 1998.3      | 45.5                | 1893.0            | 1925.0   | 1939.0   | 2057.1  | 2072.1  | 2101.0  | 1861.0    | 2137.0  |
| Admitted | 1000.0      | 0.0                 | 1000.0            | 1000.0   | 1000.0   | 1000.0  | 1000.0  | 1000.0  | 1000.0    | 1000.0  |
| Rejected | 998.3       | 45.5                | 893.0             | 925.0    | 939.0    | 1057.1  | 1072.0  | 1101.0  | 861.0     | 1137.0  |
| Outcomes | success=997 | unmet_at_capacity=3 | rejection_limit=0 |

## ppo_s1_bc_v2.zip

**Summary:** Decrease in performance.

Episodes: 1000

|          | Mean        | Std                 | P01               | P05      | P10      | P90     | P95     | P99     | Min       | Max     |
| -------- | ----------- | ------------------- | ----------------- | -------- | -------- | ------- | ------- | ------- | --------- | ------- |
| Reward   | -1091.14    | 1262.90             | -1124.01          | -1089.00 | -1073.00 | -952.90 | -935.00 | -898.98 | -21043.00 | -878.00 |
| length   | 2011.1      | 46.7                | 1899.0            | 1935.0   | 1952.9   | 2072.0  | 2087.0  | 2120.0  | 1878.0    | 2149.0  |
| Admitted | 1000.0      | 0.0                 | 1000.0            | 1000.0   | 1000.0   | 1000.0  | 1000.0  | 1000.0  | 1000.0    | 1000.0  |
| Rejected | 1011.1      | 46.7                | 899.0             | 935.0    | 952.9    | 1072.0  | 1087.0  | 1120.0  | 878.0     | 1149.0  |
| Outcomes | success=996 | unmet_at_capacity=4 | rejection_limit=0 |

## ppo_bc_s1_oracle500.zip

Trained on 500 games played perfectly by an oracle that knows all people will come during episode (using `oracle_baseline.py`). (Dataset: `s1_oracle_500.npz`).

Episodes: 1000

|          | Mean        | Std                   | P01               | P05       | P10       | P90     | P95     | P99     | Min       | Max     |
| -------- | ----------- | --------------------- | ----------------- | --------- | --------- | ------- | ------- | ------- | --------- | ------- |
| Reward   | -7885.46    | 9533.23               | -20983.03         | -20953.00 | -20930.00 | -860.00 | -845.95 | -808.99 | -21031.00 | -777.00 |
| length   | 1905.5      | 43.3                  | 1803.0            | 1833.0    | 1852.0    | 1962.0  | 1977.0  | 2004.0  | 1764.0    | 2031.0  |
| Admitted | 1000.0      | 0.0                   | 1000.0            | 1000.0    | 1000.0    | 1000.0  | 1000.0  | 1000.0  | 1000.0    | 1000.0  |
| Rejected | 905.5       | 43.3                  | 803.0             | 833.0     | 852.0     | 962.0   | 977.0   | 1004.0  | 764.0     | 1031.0  |
| Outcomes | success=651 | unmet_at_capacity=349 | rejection_limit=0 |

# Scenario 2

**Current Best (Always Win):** TBD
**Current Best (Risk On / Go For High Score):** TBD

## ppo_bc_s2_50.zip

[NEED TO RE RUN]

## ppo_s2_bc50_cur

[NEED TO RE RUN]

## ppo_bc_s2_200.zip

[NEED TO RE RUN]

## ppo_s2_bc200_cur_nv

[NEED TO RE RUN]

## ppo_s2_bc200_riskon_p90

Training Configuration:

```bash
python manage.py train_ppo --scenario 2 --init-from models/ppo_s2_bc200_cur_nv.zip --total-timesteps 1200000 --n-envs 4 --gamma 0.9997 --gae-lambda 0.997 --n-steps 8192 --ent-coef 0.02 --shape-coef 6.0 --nonhelp-penalty 1.0 --success-bonus 40000 --minmeet-bonus 2.0 --fail-penalty-scale 0.5 --success-bonus-per-saved 2.0 --late-reject-weight 0.5 --eval-freq 50000 --eval-episodes 40 --eval-percentile 90 --no-vecnorm --log-dir runs/ppo_s2_bc200_riskon_p90 --save-path models/ppo_s2_bc200_riskon_p90.zip
```

[NEED TO RE RUN]

## ppo_s2_bc200_riskon_p95_push

This model was initialized from `ppo_s2_bc200_riskon_p90` and then two training runs with different params were applied.

[NEED TO RE RUN]

## ppo_s2_bc200_riskon_p99_push

Training Configuration:

```bash
python manage.py train_ppo --scenario 2 --init-from runs/ppo_s2_bc200_riskon_p95_push/best/best_model.zip --total-timesteps 800000 --n-envs 4 --gamma 0.9997 --gae-lambda 0.997 --n-steps 8192 --ent-coef 0.05 --shape-coef 6.0 --nonhelp-penalty 0.3 --success-bonus 40000 --minmeet-bonus 4.0 --fail-penalty-scale 0.15 --success-bonus-per-saved 10.0 --late-reject-weight 1.0 --eval-freq 50000 --eval-episodes 300 --eval-percentile 99 --no-vecnorm --log-dir runs/ppo_s2_bc200_riskon_p99_push --save-path models/ppo_s2_bc200_riskon_p99_push.zip
```

[NEED TO RE RUN]

## ppo_bc_s2_oracle500

Trained on 500 games played perfectly by an oracle that knows all people will come during episode (using `oracle_baseline.py`). (Dataset: `s2_oracle_500.npz`).

[NEED TO RE RUN]

## ppo_s2_oracle_p99_push

Training Configuration:

```bash
python manage.py train_ppo --scenario 2 --init-from models/ppo_bc_s2_oracle500.zip --total-timesteps 800000 --n-envs 4 --gamma 0.9997 --gae-lambda 0.997 --n-steps 8192 --ent-coef 0.05 --shape-coef 6.0 --nonhelp-penalty 0.3 --success-bonus 40000 --minmeet-bonus 4.0 --fail-penalty-scale 0.15 --success-bonus-per-saved 10.0 --late-reject-weight 1.0 --eval-freq 50000 --eval-episodes 300 --eval-percentile 99 --no-vecnorm --log-dir runs/ppo_s2_oracle_p99_push --save-path models/ppo_s2_oracle_p99_push.zip
```

[NEED TO RE RUN]

# Scenario 3

## ppo_bc_s3_oracle500

Trained on 415\* games played perfectly by an oracle that knows all people will come during episode (using `oracle_baseline.py`). (Dataset: `s3_oracle_500.npz`). (\* We tried to generate 500, but 85 games were too complex for the solver to figure out optimal strategy before timing out.)

Episodes: 1000

|          | Mean        | Std                   | P01               | P05       | P10       | P90      | P95      | P99      | Min       | Max      |
| -------- | ----------- | --------------------- | ----------------- | --------- | --------- | -------- | -------- | -------- | --------- | -------- |
| Reward   | -10817.39   | 9128.59               | -25283.08         | -25059.15 | -24930.10 | -4666.80 | -4575.95 | -4420.94 | -25491.00 | -4273.00 |
| length   | 5877.4      | 211.5                 | 5396.0            | 5535.0    | 5609.0    | 6153.0   | 6242.1   | 6407.0   | 5273.0    | 6653.0   |
| Admitted | 1000.0      | 0.0                   | 1000.0            | 1000.0    | 1000.0    | 1000.0   | 1000.0   | 1000.0   | 1000.0    | 1000.0   |
| Rejected | 4877.4      | 211.5                 | 4396.0            | 4535.0    | 4609.0    | 5153.0   | 5242.1   | 5407.0   | 4273.0    | 5653.0   |
| Outcomes | success=703 | unmet_at_capacity=297 | rejection_limit=0 |
