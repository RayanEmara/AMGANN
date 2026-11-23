FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive

ENV OMPI_MCA_btl_vader_single_copy_mechanism=none
ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        unzip \
        wget \
        zip \
        ca-certificates \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY postInstall /
RUN /postInstall
