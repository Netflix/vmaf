FROM ubuntu
RUN apt-get update -qq

RUN apt-get install -y build-essential git
ENV TZ=UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get install -y python python-setuptools python-dev python-tk python-pip
RUN pip install --upgrade pip
RUN pip install numpy scipy matplotlib notebook pandas sympy nose scikit-learn scikit-image h5py sureal

RUN git clone --depth 1 https://github.com/Netflix/vmaf.git vmaf
WORKDIR vmaf/
ENV PYTHONPATH=/vmaf/python/src:/vmaf:$PYTHONPATH
ENV PATH=/vmaf:/vmaf/wrapper:$PATH
RUN make
WORKDIR /root/
