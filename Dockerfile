FROM ubuntu:22.04

# get and install building tools
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    ninja-build \
    nasm \
    doxygen \
    python3 \
    python3-pip \
    python3-venv \
    xxd

# retrieve source code
COPY . /vmaf

# setup environment
ENV PATH=/vmaf:/vmaf/libvmaf/build/tools:$PATH

# make vmaf
RUN cd /vmaf && make clean && make

# install python tools
RUN pip3 install --no-cache-dir -r /vmaf/python/requirements.txt

WORKDIR /vmaf

ENV PYTHONPATH=python

ENTRYPOINT [ "vmaf" ]
