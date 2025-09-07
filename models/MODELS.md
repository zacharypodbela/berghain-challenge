# Scenario 1

**Current Best (Always Win):** runs/ppo_s1_bc_v1/best/best_model.zip

## ppo_s1_bc_v1

|          |                                                   |
| -------- | ------------------------------------------------- |
| Model    | models/ppo_s1_bc_v1.zip                           |
| Scenario | 1                                                 |
| Episodes | 100                                               |
| Reward   | mean=-16004.15 std=9332.03                        |
| length   | mean=3404.2 std=102.5                             |
| Admitted | mean=1000.0 std=0.0                               |
| Rejected | mean=2404.2 std=102.5                             |
| Outcomes | success=32 unmet_at_capacity=68 rejection_limit=0 |

|          |                                                  |
| -------- | ------------------------------------------------ |
| Model    | **runs/ppo_s1_bc_v1/best/best_model.zip**        |
| Scenario | 1                                                |
| Episodes | 100                                              |
| Reward   | mean=-1190.02 std=1988.13                        |
| length   | mean=1990.0 std=43.4                             |
| Admitted | mean=1000.0 std=0.0                              |
| Rejected | mean=990.0 std=43.4                              |
| Outcomes | success=99 unmet_at_capacity=1 rejection_limit=0 |

## ppo_s1_bc_v2

|          |                                                   |
| -------- | ------------------------------------------------- |
| Model    | models/ppo_s1_bc_v2.zip                           |
| Scenario | 1                                                 |
| Episodes | 100                                               |
| Reward   | mean=-20000.00 std=0.00                           |
| length   | mean=20876.8 std=5.1                              |
| Admitted | mean=876.8 std=5.1                                |
| Rejected | mean=20000.0 std=0.0                              |
| Outcomes | success=0 unmet_at_capacity=0 rejection_limit=100 |

|          |                                                  |
| -------- | ------------------------------------------------ |
| Model    | runs/ppo_s1_bc_v2/best/best_model.zip            |
| Scenario | 1                                                |
| Episodes | 100                                              |
| Reward   | mean=-1202.57 std=1988.22                        |
| length   | mean=2002.6 std=45.1                             |
| Admitted | mean=1000.0 std=0.0                              |
| Rejected | mean=1002.6 std=45.1                             |
| Outcomes | success=99 unmet_at_capacity=1 rejection_limit=0 |

# Scenario 2

**Current Best (Always Win):** runs/ppo_s2_bc200_cur_nv/best/best_model.zip
**Current Best (Risk On / Go For High Score):** runs/ppo_s2_bc200_riskon_p99_push/best/best_model.zip

## ppo_s2_bc50_cur

|          |                                                   |
| -------- | ------------------------------------------------- |
| Model    | models/ppo_s2_bc50_cur.zip                        |
| Scenario | 2                                                 |
| Episodes | 100                                               |
| Reward   | mean=-20000.00 std=0.00                           |
| length   | mean=20828.8 std=4.6                              |
| Admitted | mean=828.8 std=4.6                                |
| Rejected | mean=20000.0 std=0.0                              |
| Outcomes | success=0 unmet_at_capacity=0 rejection_limit=100 |

|          |                                                   |
| -------- | ------------------------------------------------- |
| Model    | runs/ppo_s2_bc50_cur/best/best_model.zip          |
| Scenario | 2                                                 |
| Episodes | 100                                               |
| Reward   | mean=-5410.72 std=204.28                          |
| length   | mean=6410.7 std=204.3                             |
| Admitted | mean=1000.0 std=0.0                               |
| Rejected | mean=5410.7 std=204.3                             |
| Outcomes | success=100 unmet_at_capacity=0 rejection_limit=0 |

## ppo_s2_bc200_cur_nv

|          |                                                   |
| -------- | ------------------------------------------------- |
| Model    | models/ppo_s2_bc200_cur_nv.zip                    |
| Scenario | 2                                                 |
| Episodes | 100                                               |
| Reward   | mean=-4998.24 std=209.88                          |
| length   | mean=5998.2 std=209.9                             |
| Admitted | mean=1000.0 std=0.0                               |
| Rejected | mean=4998.2 std=209.9                             |
| Outcomes | success=100 unmet_at_capacity=0 rejection_limit=0 |

|          |                                                   |
| -------- | ------------------------------------------------- |
| Model    | **runs/ppo_s2_bc200_cur_nv/best/best_model.zip**  |
| Scenario | 2                                                 |
| Episodes | 100                                               |
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

Model: models/ppo_s2_bc200_riskon_p90.zip
Scenario: 2
Episodes: 100

|          | Mean          | Std                 | P90               | P95          |
| -------- | ------------- | ------------------- | ----------------- | ------------ |
| Reward   | mean=-4938.04 | std=204.95          | p90=-4653.50      | p95=-4629.00 |
| length   | mean=5938.0   | std=204.9           | p90=6170.7        | p95=6224.2   |
| Admitted | mean=1000.0   | std=0.0             | p90=1000.0        | p95=1000.0   |
| Rejected | mean=4938.0   | std=204.9           | p90=5170.7        | p95=5224.2   |
| Outcomes | success=100   | unmet_at_capacity=0 | rejection_limit=0 |              |

Model: runs/ppo_s2_bc200_riskon_p90/best/best_model.zip
Scenario: 2
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

Model: models/ppo_s2_bc200_riskon_p95_push.zip
Scenario: 2
Episodes: 1000

|          | Mean        | Std                 | P01               | P05      | P10      | P90      | P95      | P99      | Min       | Max      |
| -------- | ----------- | ------------------- | ----------------- | -------- | -------- | -------- | -------- | -------- | --------- | -------- |
| Reward   | -4964.22    | 657.40              | -5426.28          | -5263.00 | -5202.00 | -4691.80 | -4601.90 | -4477.91 | -24748.00 | -4349.00 |
| length   | 5944.2      | 201.0               | 5477.9            | 5601.9   | 5691.8   | 6201.1   | 6257.3   | 6415.1   | 5349.0    | 6510.0   |
| Admitted | 1000.0      | 0.0                 | 1000.0            | 1000.0   | 1000.0   | 1000.0   | 1000.0   | 1000.0   | 1000.0    | 1000.0   |
| Rejected | 4944.2      | 201.0               | 4477.9            | 4601.9   | 4691.8   | 5201.1   | 5257.3   | 5415.1   | 4349.0    | 5510.0   |
| Outcomes | success=999 | unmet_at_capacity=1 | rejection_limit=0 |

Model: runs/ppo_s2_bc200_riskon_p95_push/best/best_model.zip
Scenario: 2
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

Model: models/ppo_s2_bc200_riskon_p99_push.zip
Scenario: 2
Episodes: 100

|          | Mean      | Std                   | P01               | P05       | P10       | P90       | P95       | P99       | Min       | Max       |
| -------- | --------- | --------------------- | ----------------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- |
| Reward   | -23480.99 | 145.17                | -23793.30         | -23699.30 | -23658.50 | -23305.70 | -23247.75 | -23191.15 | -23823.00 | -23008.00 |
| length   | 4481.0    | 145.2                 | 4191.1            | 4247.8    | 4305.7    | 4658.5    | 4699.3    | 4793.3    | 4008.0    | 4823.0    |
| Admitted | 1000.0    | 0.0                   | 1000.0            | 1000.0    | 1000.0    | 1000.0    | 1000.0    | 1000.0    | 1000.0    | 1000.0    |
| Rejected | 3481.0    | 145.2                 | 3191.2            | 3247.8    | 3305.7    | 3658.5    | 3699.3    | 3793.3    | 3008.0    | 3823.0    |
| Outcomes | success=0 | unmet_at_capacity=100 | rejection_limit=0 |

Model: **runs/ppo_s2_bc200_riskon_p99_push/best/best_model.zip**
Scenario: 2
Episodes: 100

|          | Mean       | Std                  | P01               | P05       | P10       | P90       | P95      | P99      | Min       | Max      |
| -------- | ---------- | -------------------- | ----------------- | --------- | --------- | --------- | -------- | -------- | --------- | -------- |
| Reward   | -21797.46  | 6032.77              | -24177.25         | -24043.05 | -23980.80 | -21531.00 | -3711.90 | -3578.12 | -24202.00 | -3293.00 |
| length   | 4797.5     | 161.4                | 4476.1            | 4535.6    | 4554.0    | 5008.7    | 5043.1   | 5177.2   | 4293.0    | 5202.0   |
| Admitted | 1000.0     | 0.0                  | 1000.0            | 1000.0    | 1000.0    | 1000.0    | 1000.0   | 1000.0   | 1000.0    | 1000.0   |
| Rejected | 3797.5     | 161.4                | 3476.2            | 3535.6    | 3554.0    | 4008.7    | 4043.1   | 4177.2   | 3293.0    | 4202.0   |
| Outcomes | success=10 | unmet_at_capacity=90 | rejection_limit=0 |

## ppo_bc_s2_oracle500.zip

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

Episodes: 100

|          | Mean       | Std                  | P01               | P05       | P10       | P90       | P95      | P99      | Min       | Max      |
| -------- | ---------- | -------------------- | ----------------- | --------- | --------- | --------- | -------- | -------- | --------- | -------- |
| Reward   | -21641.85  | 5999.13              | -24067.21         | -23907.50 | -23843.60 | -21372.10 | -3655.25 | -3362.92 | -24088.00 | -3256.00 |
| length   | 4641.9     | 172.5                | 4301.5            | 4363.9    | 4402.8    | 4871.5    | 4927.6   | 5067.2   | 4256.0    | 5088.0   |
| Admitted | 1000.0     | 0.0                  | 1000.0            | 1000.0    | 1000.0    | 1000.0    | 1000.0   | 1000.0   | 1000.0    | 1000.0   |
| Rejected | 3641.8     | 172.5                | 3301.5            | 3363.9    | 3402.8    | 3871.5    | 3927.6   | 4067.2   | 3256.0    | 4088.0   |
| Outcomes | success=10 | unmet_at_capacity=90 | rejection_limit=0 |
