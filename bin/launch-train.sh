#!/bin/bash

# entire script fails if a single command fails
set -e

# script should be run from the project directory
export PROJECT_DIR="$PWD"

# creates a separate directory for each job
JOB_NAME=example-training-job
mkdir -p "$PROJECT_DIR"/results/"$JOB_NAME"

# launch the training job
CPUS_PER_GPU=6
sbatch --job-name "$JOB_NAME" --cpus-per-gpu $CPUS_PER_GPU \
    "$PROJECT_DIR"/bin/train.sbatch "$PROJECT_DIR"/src/train-argparse.py \
        --dataloader-num-workers $CPUS_PER_GPU \
        --data-dir data/ \
        --output-dir results/$JOB_NAME/ \
        --tqdm-disable

