# pytorch-gpu-data-science-project

Repository containing scaffolding for a Python 3-based data science project with GPU acceleration using the [PyTorch](https://pytorch.org/) ecosystem. 

## Creating a new project from this template

Simply follow the [instructions](https://help.github.com/en/articles/creating-a-repository-from-a-template) to create a new project repository from this template.

## Project organization

Project organization is based on ideas from [_Good Enough Practices for Scientific Computing_](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005510).

1. Put each project in its own directory, which is named after the project.
2. Put external scripts or compiled programs in the `bin` directory.
3. Put raw data and metadata in a `data` directory.
4. Put text documents associated with the project in the `doc` directory.
5. Put all Docker related files in the `docker` directory.
6. Install the Conda environment into an `env` directory. 
7. Put all notebooks in the `notebooks` directory.
8. Put files generated during cleanup and analysis in a `results` directory.
9. Put project source code in the `src` directory.
10. Name all files to reflect their content or function.

## Building the Conda environment

After adding any necessary dependencies that should be downloaded via `conda` to the `environment.yml` file 
and any dependencies that should be downloaded via `pip` to the `requirements.txt` file you create the 
Conda environment in a sub-directory `./env`of your project directory by running the following commands.

```bash
$ export ENV_PREFIX=$PWD/env
$ export HOROVOD_CUDA_HOME=$ENV_PREFIX
$ export HOROVOD_NCCL_HOME=$ENV_PREFIX
$ export HOROVOD_GPU_ALLREDUCE=NCCL 
$ conda env create --prefix $ENV_PREFIX --file environment.yml
```

Once the new environment has been created you can activate the environment with the following 
command.

```bash
$ conda activate $ENV_PREFIX
(/path/to/env) $
```

Note that the `ENV_PREFIX` directory is *not* under version control as it can always be re-created as 
necessary.

### Building JupyterLab extensions (optional)

If you wish to use any JupyterLab extensions included in the `environment.yml` and `requirements.txt` 
files then you need to activate the environment and rebuild the JupyterLab application using the 
following commands to source the `postBuild` script.

```bash
$ conda activate $ENV_PREFIX # optional if environment already active
(/path/to/env) $ . postBuild
```

### Verifying the Conda environment

After building the Conda environment you can check that Horovod has been built with support for 
TensorFlow and MPI with the following command.

```bash
$ conda activate $ENV_PREFIX # optional if environment already active
(/path/to/env) $ horovodrun --check-build
```

You should see output similar to the following.

```
Horovod v0.18.2:

Available Frameworks:
    [ ] TensorFlow
    [X] PyTorch
    [ ] MXNet

Available Controllers:
    [X] MPI
    [X] Gloo

Available Tensor Operations:
    [X] NCCL
    [ ] DDL
    [ ] MLSL
    [X] MPI
    [X] Gloo  
```

### Listing the full contents of the Conda environment

The list of explicit dependencies for the project are listed in the `environment.yml` file. To see 
the full lost of packages installed into the environment run the following command.

```bash
conda list --prefix $ENV_PREFIX
```

### Updating the Conda environment

If you add (remove) dependencies to (from) the `environment.yml` file or the `requirements.txt` file 
after the environment has already been created, then you can re-create the environment with the 
following command.

```bash
$ conda env create --prefix $ENV_PREFIX --file environment.yml --force
```

## Using Docker

In order to build Docker images for your project and run containers with GPU acceleration you will 
need to install 
[Docker](https://docs.docker.com/install/linux/docker-ce/ubuntu/), 
[Docker Compose](https://docs.docker.com/compose/install/) and the 
[NVIDIA Docker runtime](https://github.com/NVIDIA/nvidia-docker).

Detailed instructions for using Docker to build and image and launch containers can be found in 
the `docker/README.md`.
