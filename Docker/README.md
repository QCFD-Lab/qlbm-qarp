# Container files

We provide a docker file for `qlbm` development.

## Development container

`build_cpu.Dockerfile` can be used to build a container image that has all
dependencies of `qlbm` for active development.

```bash
docker build -t qlbm -f Docker/build_cpu.Dockerfile .
```

Once the image is built, it will have all the dependencies, but not `qlbm`
itself. You can actively link the repository files to the running container
with the command

```bash
docker run -it --volume $(pwd):/qlbm/ qlbm
```

Once inside the container, you can quickly link the files in the volume to
`pip` with

```
pip install -e .[dev,docs]
```

This will allow you to edit the files on your machine's file system in a text
editor of your choice, and have the updates immediately available in the
running container, without reinstalling anything.

## GPU container

There is no GPU container.
