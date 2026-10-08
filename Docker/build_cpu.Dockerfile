FROM python:3.13-bookworm

RUN apt-get update -qq && apt-get install -y -qq pandoc libglfw3-dev libgles2-mesa-dev

WORKDIR /qlbm

COPY pyproject.toml pyproject.toml
RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir -e ".[dev,docs]" \
    && rm -rf /root/.cache/pip

ENTRYPOINT /bin/bash
