FROM ubuntu:22.04
ARG NV_CODEC_TAG="876af32a202d0de83bd1d36fe74ee0f7fcf86b0d"

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
    xxd \
    clang \
    wget \
    unzip \
    nvidia-cuda-dev \
    nvidia-cuda-toolkit

# retrieve source code
COPY . /vmaf

# setup environment
ENV PATH=/vmaf:/vmaf/libvmaf/build/tools:$PATH

RUN wget https://github.com/FFmpeg/nv-codec-headers/archive/${NV_CODEC_TAG}.zip && unzip ${NV_CODEC_TAG}.zip && cd nv-codec-headers-${NV_CODEC_TAG} && make && make install

# make vmaf
# when disabling NVCC, libvmaf will be built without cubin's which will compile kernels at start of the container
RUN cd /vmaf && make clean && make ENABLE_NVCC=true && make install

# install python tools
RUN pip3 install --no-cache-dir -r /vmaf/python/requirements.txt

WORKDIR /vmaf

ENV PYTHONPATH=python

ENTRYPOINT [ "vmaf" ]
