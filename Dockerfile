FROM ubuntu:18.04

# setup timezone
ENV TZ=UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# get and install building tools
RUN apt-get update && \
	apt-get install -y --no-install-recommends \
        build-essential \
        git \
        ninja-build \
        python3 \
        python3-dev \
        python3-pip \
        python3-setuptools \
        python3-tk \
        && \
	apt-get clean && \
	rm -rf /var/lib/apt/lists

# get and install tools for python
RUN pip3 install --upgrade pip
RUN pip install numpy scipy matplotlib notebook pandas sympy nose scikit-learn scikit-image h5py sureal meson

# retrieve source code
COPY . /vmaf

# setup environment
ENV PYTHONPATH=/vmaf/python/src:/vmaf:$PYTHONPATH
ENV PATH=/vmaf:/vmaf/src/libvmaf:$PATH

# make vmaf
RUN cd /vmaf && make

WORKDIR /root/
