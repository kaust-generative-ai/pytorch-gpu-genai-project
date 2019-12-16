FROM nvidia/cuda:10.1-base-ubuntu16.04

LABEL maintainer="pughdr <david.pugh@kaust.edu.sa>"

SHELL [ "/bin/bash", "-c" ]

RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 curl git gcc && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create a non-root user
ARG username=al-khawarizmi
ARG uid=1000
ARG gid=100
ENV USER ${username}
ENV UID ${uid}
ENV GID ${gid}
ENV HOME /home/$USER

RUN adduser --disabled-password \
    --gecos "Non-root user" \
    --uid $UID \
    --gid $GID \
    --home $HOME \
    $USER

# Dockerhub not yet supporting COPY --chown $UID:$GID syntax
COPY environment.yml /tmp/environment.yml
RUN chown $UID:$GID /tmp/environment.yml

COPY postBuild /usr/local/postBuild.sh
RUN chown $UID:$GID /usr/local/postBuild.sh && \
    chmod +x /usr/local/postBuild.sh

COPY docker/entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chown $UID:$GID /usr/local/bin/docker-entrypoint.sh && \
    chmod +x /usr/local/bin/docker-entrypoint.sh

# install Miniconda as non-root user
USER $USER

ENV MINICONDA_VERSION 4.7.12
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh -O ~/miniconda.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b && \
    rm ~/miniconda.sh && \
    echo ". ${HOME}/miniconda3/etc/profile.d/conda.sh" >> ~/.profile

RUN mkdir $HOME/app
WORKDIR $HOME/app

ENV CONDA_PREFIX ./env
RUN ~/miniconda3/bin/conda update --name base --channel defaults conda && \
    ~/miniconda3/bin/conda env create --prefix $CONDA_PREFIX --file /tmp/environment.yml && \
    ~/miniconda3/bin/conda clean --all --yes

ENV HOROVOD_CUDA_HOME $CONDA_PREFIX
ENV HOROVOD_NCCL_HOME $CONDA_PREFX
ENV HOROVOD_GPU_ALLREDUCE NCCL
RUN source ~/miniconda3/bin/activate $CONDA_PREFIX && \
    /usr/local/postBuild.sh && \
    pip install horovod==0.18.* --no-cache-dir && \
    source ~/miniconda3/bin/deactivate

# use an entrypoint script to insure conda environment is properly activated at runtime
ENTRYPOINT [ "/usr/local/bin/docker-entrypoint.sh" ]

# default command will be to launch JupyterLab server for development
CMD [ "jupyter", "lab", "--no-browser", "--ip", "0.0.0.0" ]
