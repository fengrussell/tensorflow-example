#!/bin/bash

python -u mnist_dist_mts_ex.py \
  --batch_size=8 \
  --max_number_of_steps=10 \
  --ps_hosts=10.231.56.243:2222 \
  --worker_hosts=10.231.56.243:2223 \
  --num_gpus=1 \
  --job_name=$1 \
  --task_id=$2
