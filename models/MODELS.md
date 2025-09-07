# Scenario 1

**Current Best (Always Win):** TBD
**Current Best (Risk On / Go For High Score):** TBD

## ppo_bc_s1.zip

[NEED TO RE RUN]

## ppo_s1_bc_v1.zip

[NEED TO RE RUN]

## ppo_s1_bc_v2.zip

[NEED TO RE RUN]

## ppo_bc_s1_oracle500

[NEED TO RE RUN]

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

[NEED TO RE RUN]

## ppo_s2_oracle_p99_push

Training Configuration:

```bash
python manage.py train_ppo --scenario 2 --init-from models/ppo_bc_s2_oracle500.zip --total-timesteps 800000 --n-envs 4 --gamma 0.9997 --gae-lambda 0.997 --n-steps 8192 --ent-coef 0.05 --shape-coef 6.0 --nonhelp-penalty 0.3 --success-bonus 40000 --minmeet-bonus 4.0 --fail-penalty-scale 0.15 --success-bonus-per-saved 10.0 --late-reject-weight 1.0 --eval-freq 50000 --eval-episodes 300 --eval-percentile 99 --no-vecnorm --log-dir runs/ppo_s2_oracle_p99_push --save-path models/ppo_s2_oracle_p99_push.zip
```

[NEED TO RE RUN]
