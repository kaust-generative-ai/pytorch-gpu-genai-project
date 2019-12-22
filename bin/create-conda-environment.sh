# make sure that CUDA and NCCL are available
export ENV_PREFIX=$PWD/env
export HOROVOD_CUDA_HOME=$ENV_PREFIX
export HOROVOD_NCCL_HOME=$ENV_PREFIX
export HOROVOD_GPU_ALLREDUCE=NCCL

# create the conda environment
conda env create --prefix $ENV_PREFIX --file environment.yml --force
conda activate $ENV_PREFIX
pip install --requirement requirements.txt

# source postBuild to enable JupyterLab extensions
. postBuild
