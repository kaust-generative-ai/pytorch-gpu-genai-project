#!/bin/bash --login
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=v100:1
#SBATCH --cpus-per-gpu=6  
#SBATCH --mem=64G 
#SBATCH --partition=debug 
#SBATCH --mail-type=ALL
#SBATCH --output=results/%x/%j-slurm.out
#SBATCH --error=results/%x/%j-slurm.err

# entire script fails if single command fails
set -e

# activate the conda environment
module purge
ENV_PREFIX="$PROJECT_DIR"/env
conda activate $ENV_PREFIX

# use srun to launch NVDashboard server in order to reserve a port
srun --resv-ports=1 ./bin/launch-nvdashboard-server.srun &
NVDASHBOARD_PID=$!

# launch the training script
python "$@"

# shutdown the NVDashboard server
kill $NVDASHBOARD_PID
