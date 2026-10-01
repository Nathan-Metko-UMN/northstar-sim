# Northstar-CV's dev image (either JetPack's) plus socat, which turns the harness's TCP UART into a
# local PTY. Built by `nssim build-cv` (or, from this directory:
# docker build -t northstar-cv:sim-jetpack7 --build-arg BASE=northstar-cv:jetpack7 -f cv-sim.Dockerfile .)
ARG BASE=northstar-cv:jetpack7
FROM ${BASE}
USER root
RUN apt-get update && apt-get install -y --no-install-recommends socat && rm -rf /var/lib/apt/lists/*
USER ubuntu
