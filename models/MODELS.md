# Scenario 1

**Current Best:** runs/ppo_s1_bc_v1/best/best_model.zip

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

**Current Best:** runs/ppo_s2_bc200_cur_nv/best/best_model.zip

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
