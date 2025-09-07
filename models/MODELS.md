# Scenario 1

**Current Best (Always Win):** `ppo_s1_bc_v1.zip`
**Current Best (Risk On / Go For High Score):** `ppo_bc_s1_oracle500.zip`

## ppo_s1_bc_v1.zip

Episodes: 100

|          |                                                  |
| -------- | ------------------------------------------------ |
| Reward   | mean=-1190.02 std=1988.13                        |
| length   | mean=1990.0 std=43.4                             |
| Admitted | mean=1000.0 std=0.0                              |
| Rejected | mean=990.0 std=43.4                              |
| Outcomes | success=99 unmet_at_capacity=1 rejection_limit=0 |

## ppo_s1_bc_v2.zip

Episodes: 100

|          |                                                  |
| -------- | ------------------------------------------------ |
| Reward   | mean=-1202.57 std=1988.22                        |
| length   | mean=2002.6 std=45.1                             |
| Admitted | mean=1000.0 std=0.0                              |
| Rejected | mean=1002.6 std=45.1                             |
| Outcomes | success=99 unmet_at_capacity=1 rejection_limit=0 |

## ppo_bc_s1_oracle500

Episodes: 1000

|          | Mean        | Std                   | P01               | P05       | P10       | P90     | P95     | P99     | Min       | Max     |
| -------- | ----------- | --------------------- | ----------------- | --------- | --------- | ------- | ------- | ------- | --------- | ------- |
| Reward   | -9476.80    | 9898.32               | -20980.00         | -20950.00 | -20927.00 | -855.00 | -838.00 | -807.98 | -21026.00 | -773.00 |
| length   | 1896.8      | 43.5                  | 1794.9            | 1826.0    | 1841.0    | 1952.0  | 1970.0  | 1995.0  | 1762.0    | 2026.0  |
| Admitted | 1000.0      | 0.0                   | 1000.0            | 1000.0    | 1000.0    | 1000.0  | 1000.0  | 1000.0  | 1000.0    | 1000.0  |
| Rejected | 896.8       | 43.5                  | 794.9             | 826.0     | 841.0     | 952.0   | 970.0   | 995.0   | 762.0     | 1026.0  |
| Outcomes | success=571 | unmet_at_capacity=429 | rejection_limit=0 |

## Next Steps:

- `ppo_bc_s1_oracle500` could be riskier

# Scenario 2

**Current Best (Always Win):** ppo_s2_bc200_cur_nv.zip
**Current Best (Risk On / Go For High Score):** ppo_s2_bc200_riskon_p99_push.zip

## ppo_s2_bc50_cur

Episodes: 100

|          |                                                   |
| -------- | ------------------------------------------------- |
| Reward   | mean=-5410.72 std=204.28                          |
| length   | mean=6410.7 std=204.3                             |
| Admitted | mean=1000.0 std=0.0                               |
| Rejected | mean=5410.7 std=204.3                             |
| Outcomes | success=100 unmet_at_capacity=0 rejection_limit=0 |

## ppo_s2_bc200_cur_nv

Episodes: 100

|          |                                                   |
| -------- | ------------------------------------------------- |
| Reward   | mean=-4996.20 std=209.97                          |
| length   | mean=5996.2 std=210.0                             |
| Admitted | mean=1000.0 std=0.0                               |
| Rejected | mean=4996.2 std=210.0                             |
| Outcomes | success=100 unmet_at_capacity=0 rejection_limit=0 |

## ppo_s2_bc200_riskon_p90

Training Configuration:

```bash
python manage.py train_ppo --scenario 2 --init-from models/ppo_s2_bc200_cur_nv.zip --total-timesteps 1200000 --n-envs 4 --gamma 0.9997 --gae-lambda 0.997 --n-steps 8192 --ent-coef 0.02 --shape-coef 6.0 --nonhelp-penalty 1.0 --success-bonus 40000 --minmeet-bonus 2.0 --fail-penalty-scale 0.5 --success-bonus-per-saved 2.0 --late-reject-weight 0.5 --eval-freq 50000 --eval-episodes 40 --eval-percentile 90 --no-vecnorm --log-dir runs/ppo_s2_bc200_riskon_p90 --save-path models/ppo_s2_bc200_riskon_p90.zip
```

Episodes: 100

|          | Mean          | Std                 | P90               | P95          |
| -------- | ------------- | ------------------- | ----------------- | ------------ |
| Reward   | mean=-4948.35 | std=204.82          | p90=-4669.40      | p95=-4637.60 |
| length   | mean=5948.4   | std=204.8           | p90=6176.2        | p95=6227.4   |
| Admitted | mean=1000.0   | std=0.0             | p90=1000.0        | p95=1000.0   |
| Rejected | mean=4948.4   | std=204.8           | p90=5176.2        | p95=5227.4   |
| Outcomes | success=100   | unmet_at_capacity=0 | rejection_limit=0 |              |

## ppo_s2_bc200_riskon_p95_push

This model was initialized from `ppo_s2_bc200_riskon_p90` and then two training runs with different params were applied.

Episodes: 1000

|          | Mean        | Std                 | P01               | P05      | P10      | P90      | P95      | P99      | Min       | Max      |
| -------- | ----------- | ------------------- | ----------------- | -------- | -------- | -------- | -------- | -------- | --------- | -------- |
| Reward   | -4968.83    | 657.28              | -5427.27          | -5270.55 | -5206.10 | -4693.00 | -4602.95 | -4478.00 | -24748.00 | -4349.00 |
| length   | 5948.8      | 201.1               | 5478.0            | 5602.9   | 5693.0   | 6206.0   | 6269.1   | 6415.1   | 5349.0    | 6517.0   |
| Admitted | 1000.0      | 0.0                 | 1000.0            | 1000.0   | 1000.0   | 1000.0   | 1000.0   | 1000.0   | 1000.0    | 1000.0   |
| Rejected | 4948.8      | 201.1               | 4478.0            | 4602.9   | 4693.0   | 5206.0   | 5269.1   | 5415.1   | 4349.0    | 5517.0   |
| Outcomes | success=999 | unmet_at_capacity=1 | rejection_limit=0 |

## ppo_s2_bc200_riskon_p99_push

Training Configuration:

```bash
python manage.py train_ppo --scenario 2 --init-from runs/ppo_s2_bc200_riskon_p95_push/best/best_model.zip --total-timesteps 800000 --n-envs 4 --gamma 0.9997 --gae-lambda 0.997 --n-steps 8192 --ent-coef 0.05 --shape-coef 6.0 --nonhelp-penalty 0.3 --success-bonus 40000 --minmeet-bonus 4.0 --fail-penalty-scale 0.15 --success-bonus-per-saved 10.0 --late-reject-weight 1.0 --eval-freq 50000 --eval-episodes 300 --eval-percentile 99 --no-vecnorm --log-dir runs/ppo_s2_bc200_riskon_p99_push --save-path models/ppo_s2_bc200_riskon_p99_push.zip
```

Episodes: 100

|          | Mean       | Std                  | P01               | P05       | P10       | P90       | P95      | P99      | Min       | Max      |
| -------- | ---------- | -------------------- | ----------------- | --------- | --------- | --------- | -------- | -------- | --------- | -------- |
| Reward   | -21797.46  | 6032.77              | -24177.25         | -24043.05 | -23980.80 | -21531.00 | -3711.90 | -3578.12 | -24202.00 | -3293.00 |
| length   | 4797.5     | 161.4                | 4476.1            | 4535.6    | 4554.0    | 5008.7    | 5043.1   | 5177.2   | 4293.0    | 5202.0   |
| Admitted | 1000.0     | 0.0                  | 1000.0            | 1000.0    | 1000.0    | 1000.0    | 1000.0   | 1000.0   | 1000.0    | 1000.0   |
| Rejected | 3797.5     | 161.4                | 3476.2            | 3535.6    | 3554.0    | 4008.7    | 4043.1   | 4177.2   | 3293.0    | 4202.0   |
| Outcomes | success=10 | unmet_at_capacity=90 | rejection_limit=0 |

## ppo_bc_s2_oracle500

Episodes: 1000

|          | Mean        | Std                   | P01               | P05       | P10       | P90      | P95      | P99      | Min       | Max      |
| -------- | ----------- | --------------------- | ----------------- | --------- | --------- | -------- | -------- | -------- | --------- | -------- |
| Reward   | -20101.73   | 7836.39               | -24297.03         | -24158.00 | -24087.20 | -3892.30 | -3765.95 | -3557.90 | -24558.00 | -3393.00 |
| length   | 4881.7      | 177.2                 | 4498.9            | 4599.0    | 4657.8    | 5113.0   | 5177.2   | 5300.2   | 4389.0    | 5558.0   |
| Admitted | 1000.0      | 0.0                   | 1000.0            | 1000.0    | 1000.0    | 1000.0   | 1000.0   | 1000.0   | 1000.0    | 1000.0   |
| Rejected | 3881.7      | 177.2                 | 3498.9            | 3599.0    | 3657.8    | 4113.0   | 4177.2   | 4300.2   | 3389.0    | 4558.0   |
| Outcomes | success=189 | unmet_at_capacity=811 | rejection_limit=0 |

## ppo_s2_oracle_p99_push

Training Configuration:

```bash
python manage.py train_ppo --scenario 2 --init-from models/ppo_bc_s2_oracle500.zip --total-timesteps 800000 --n-envs 4 --gamma 0.9997 --gae-lambda 0.997 --n-steps 8192 --ent-coef 0.05 --shape-coef 6.0 --nonhelp-penalty 0.3 --success-bonus 40000 --minmeet-bonus 4.0 --fail-penalty-scale 0.15 --success-bonus-per-saved 10.0 --late-reject-weight 1.0 --eval-freq 50000 --eval-episodes 300 --eval-percentile 99 --no-vecnorm --log-dir runs/ppo_s2_oracle_p99_push --save-path models/ppo_s2_oracle_p99_push.zip
```

Episodes: 1000

|          | Mean       | Std                   | P01               | P05       | P10       | P90       | P95      | P99      | Min       | Max      |
| -------- | ---------- | --------------------- | ----------------- | --------- | --------- | --------- | -------- | -------- | --------- | -------- |
| Reward   | -22567.19  | 4485.48               | -24048.06         | -23911.00 | -23849.20 | -23350.90 | -3991.95 | -3441.87 | -24365.00 | -3254.00 |
| length   | 4627.2     | 176.9                 | 4253.9            | 4350.9    | 4400.7    | 4854.0    | 4919.0   | 5048.1   | 4076.0    | 5365.0   |
| Admitted | 1000.0     | 0.0                   | 1000.0            | 1000.0    | 1000.0    | 1000.0    | 1000.0   | 1000.0   | 1000.0    | 1000.0   |
| Rejected | 3627.2     | 176.9                 | 3253.9            | 3350.9    | 3400.7    | 3854.0    | 3919.0   | 4048.1   | 3076.0    | 4365.0   |
| Outcomes | success=53 | unmet_at_capacity=947 | rejection_limit=0 |
