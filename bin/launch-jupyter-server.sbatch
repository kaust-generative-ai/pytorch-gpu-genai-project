#!/bin/bash --login
#SBATCH --time 12:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=4  
#SBATCH --mem-per-gpu=48G 
#SBATCH --partition=batch 
#SBATCH --job-name=launch-jupyter-server
#SBATCH --mail-type=ALL
#SBATCH --output=bin/%x-%j-slurm.out
#SBATCH --error=bin/%x-%j-slurm.err

# setup the environment
module purge
conda activate ./env 

# setup ssh tunneling 
export XDG_RUNTIME_DIR=/tmp IBEX_NODE=$(hostname -s) 
KAUST_USER=$(whoami) 
JUPYTERLAB_PORT=8888 
 
echo "
To connect to the compute node ${IBEX_NODE} on Ibex running your JupyterLab server, 
you need to create an ssh tunnel from your local machine to glogin node on Ibex 
using the following command.

ssh -L ${JUPYTERLAB_PORT}:${IBEX_NODE}:${JUPYTERLAB_PORT} ${KAUST_USER}@glogin.ibex.kaust.edu.sa 

Next, you need to copy the second url provided below and paste it into the browser 
on your local machine.
" >&2

# launch jupyter server 
jupyter ${1:-lab} --no-browser --port=${JUPYTERLAB_PORT} --ip=${IBEX_NODE} 
