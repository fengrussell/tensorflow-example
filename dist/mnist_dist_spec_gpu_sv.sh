#!/bin/bash

python -u mnist_dist_spec_gpu_sv.py \
  --batch_size=8 \
  --max_number_of_steps=10 \
  --ps_hosts=10.231.56.190:2222 \
  --worker_hosts=10.231.56.197:2222,10.231.56.199:2222 \
  --num_gpus=1 \
  --job_name=$1 \
  --task_id=$2
