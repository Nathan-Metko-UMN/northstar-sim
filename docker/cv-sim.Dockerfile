# Northstar-CV dev image plus socat, which turns the harness's TCP UART into a local PTY.
# Build (from this directory):  docker build -t northstar-cv:sim -f cv-sim.Dockerfile .
ARG BASE=northstar-cv:dev
FROM ${BASE}
USER root
RUN apt-get update && apt-get install -y --no-install-recommends socat && rm -rf /var/lib/apt/lists/*
USER ubuntu
