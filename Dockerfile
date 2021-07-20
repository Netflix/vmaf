FROM ubuntu:20.04

# setup timezone
ENV TZ=UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# get and install building tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    git \
    ninja-build \
    nasm \
    doxygen \
    python3 \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    python3-tk \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists

# retrieve source code
COPY . /vmaf

# install python requirements
RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir meson cython numpy

# setup environment
ENV PATH=/vmaf:/vmaf/libvmaf/build/tools:$PATH

# make vmaf
RUN cd /vmaf && make clean && make

# install python tools
RUN pip3 install --no-cache-dir -r /vmaf/python/requirements.txt

WORKDIR /vmaf

ENV PYTHONPATH=python

ENTRYPOINT [ "./python/vmaf/script/run_vmaf.py" ]
