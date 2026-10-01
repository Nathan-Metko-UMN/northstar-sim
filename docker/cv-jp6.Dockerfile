# Northstar-CV's build environment with JetPack 6's CUDA and OS: CUDA 12.6 on Ubuntu 22.04, as on
# JetPack 6.1/6.2. It compiles the same code as the Jetson (here on x86, also for its sm_87), so a
# branch can be checked against JetPack 6 as well as JetPack 7 (the dev image, CUDA 13) before it goes
# to the robot, and the build runs in the sim like the other. Includes socat for running in the sim.
# Built by `nssim build-cv --jetpack 6`.
FROM nvidia/cuda:12.6.3-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=America/Chicago \
    CC=clang \
    CXX=clang++ \
    CMAKE_GENERATOR=Ninja \
    CMAKE_C_COMPILER_LAUNCHER=ccache \
    CMAKE_CXX_COMPILER_LAUNCHER=ccache \
    CCACHE_DIR=/ws/.ccache

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        libyaml-cpp-dev \
        libeigen3-dev \
        libboost-all-dev \
        libasio-dev \
        libfastcdr-dev \
        libopencv-dev \
        zlib1g-dev \
        clang \
        libomp-dev \
        ccache \
        ninja-build \
        python3-pip \
        socat \
        curl \
        gnupg \
        ca-certificates && \
    pip3 install --upgrade cmake && \
    rm -rf /var/lib/apt/lists/*

# Northstar-CV requires librealsense2 (find_package(realsense2 REQUIRED)).
RUN mkdir -p /etc/apt/keyrings && \
    curl -fsSL https://librealsense.realsenseai.com/Debian/librealsenseai.asc | gpg --dearmor -o /etc/apt/keyrings/librealsenseai.gpg && \
    echo "deb [signed-by=/etc/apt/keyrings/librealsenseai.gpg] https://librealsense.realsenseai.com/Debian/apt-repo jammy main" \
        > /etc/apt/sources.list.d/librealsense.list && \
    apt-get update && apt-get install -y --no-install-recommends librealsense2-dev && \
    rm -rf /var/lib/apt/lists/*

RUN useradd -m -s /bin/bash -u 1000 -U ubuntu
USER ubuntu
WORKDIR /ws
