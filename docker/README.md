## Building the Docker image

The following command builds a Docker image with a particular `$IMAGE_NAME` and `$IMAGE_TAG` using the `Dockerfile`. Note that the `Dockerfile` uses the `environment.yml` file in the project root directory to create a Conda environment inside the image.

```bash
$ docker build \
  --build-arg username=$USER \
  --build-arg uid=$UID \
  --build-arg gid=$GID \
  --build-arg environment=environment.yml \
  --build-arg entrypoint=docker/entrypoint.sh \
  --file docker/Dockerfile \
  --tag $IMAGE_NAME:$IMAGE_TAG \
  ../
```

## Running the Docker image

Once you have built the image, the following command will run a container based on the image `$IMAGE_NAME:$IMAGE_TAG`.

```bash
$ docker container run \
  --rm \
  --tty \
  --volume ../bin:/home/$USER/app/bin \
  --volume ../data:/home/$USER/app/data \ 
  --volume ../doc:/home/$USER/app/doc \
  --volume ../notebooks:/home/$USER/app/notebooks \
  --volume ../results:/home/$USER/app/results \
  --volume ../src:/home/$USER/app/src \
  --runtime nvidia \
  --publish 8888:8888 \
  $IMAGE_NAME:$IMAGE_TAG
```

## Using Docker Compose

It is quite easy to make a typo whilst writing the above docker commands by hand, a less error-prone approach is to use [Docker Compose](https://docs.docker.com/compose/). The above docker commands have been encapsulated into the `docker-compose.yml` configuration file and the following command can be used to bring up a container based on our image.

```bash
$ docker-compose up --build
```

When you are done developing, the following command tears down the networking inrastructure for the running container.

```bash
$ docker-compose down
```
