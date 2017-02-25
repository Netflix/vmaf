FROM        ubuntu
RUN apt-get update -qq && apt-get install -y \
      build-essential git gcc python-numpy python-scipy python-matplotlib \
      ipython ipython-notebook python-pandas python-sympy python-nose \
      python2.7 python-setuptools python-dev
RUN easy_install pip
RUN pip install --upgrade scikit-learn
RUN pip install --upgrade h5py
RUN git clone --depth 1 https://github.com/Netflix/vmaf.git vmaf
ENV PYTHONPATH=/vmaf/python/src:$PYTHONPATH
ENV PYTHONPATH=/vmaf:$PYTHONPATH
ENV PATH=/vmaf:/vmaf/wrapper:$PATH
RUN cd /vmaf && make
# For the UserWarning: Matplotlib is building ... fc-list please see https://github.com/matplotlib/matplotlib/issues/5836
