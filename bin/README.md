# Slurm

## Interactive jobs

### Launching Jupyter server for interactive work

The job script `launch-jupyter-server.sbatch` launches a [Jupyter](https://jupyter.org/) server for 
interactive prototyping. To launch a [JupyterLab](https://jupyterlab.readthedocs.io/en/stable/) 
server use `sbatch` to submit the job script by running the following command from the project root 
directory.

```bash
sbatch --user-mail $KAUST_EMAIL ./bin/launch-jupyter-server.sbatch
```

If you prefer the classic Jupyter Notebook interface, then you can launch the Jupyter notebook 
server with the following command in the project root directory.

```bash
sbatch --user-mail $KAUST_EMAIL	./bin/launch-jupyter-server.sbatch notebook
```

Once the job has started, you can inspect the `launch-jupyter-server-$SLURM_JOB_ID-slurm.err` 
file where you will find instructions on how to access the server running in your local 
browser.

#### SSH tunneling between your local machine and Ibex compute node(s)

To connect to the compute node on Ibex running your Jupyter server, you need to create 
an ssh tunnel from your local machine to glogin node on Ibex using the following command.

```bash
ssh -L ${JUPYTERLAB_PORT}:${IBEX_NODE}:${JUPYTERLAB_PORT} ${KAUST_USER}@glogin.ibex.kaust.edu.sa
```

The exact command for your job can be copied from the 
`launch-jupyter-server-$SLURM_JOB_ID-slurm.err` file.

#### Accessing Jupyter server from your local machine

Once you have set up your SSH tunnel, in order to access the Jupyter server from your local 
machine you need to copy the second url provided in the Jupyter server logs in the 
`launch-jupyter-server-$SLURM_JOB_ID-slurm.err` file and paste it into the browser on your local machine.

