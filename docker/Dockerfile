FROM mosaicml/pytorch

WORKDIR /data-dedupe

RUN git clone https://github.com/growlix/streaming.git && \
    apt-get install vim -y && \
    cd streaming && \
    git switch deduplication && \
    pip install --user -e .[all] && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
    source "$HOME/.cargo/env" && \
    cd scripts/deduplication/exact/ && \
    cargo build && \
    pip install cmake && \
    cd /tmp && \
    wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB && \
    apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB && \
    sh -c 'echo deb https://apt.repos.intel.com/mkl all main > /etc/apt/sources.list.d/intel-mkl.list' && \
    apt-get update && \
    apt-get install intel-mkl-64bit-2018.2-046 -y && \
    update-alternatives --install /usr/lib/x86_64-linux-gnu/libblas.so     libblas.so-x86_64-linux-gnu      /opt/intel/mkl/lib/intel64/libmkl_rt.so 150 && \
    update-alternatives --install /usr/lib/x86_64-linux-gnu/libblas.so.3   libblas.so.3-x86_64-linux-gnu    /opt/intel/mkl/lib/intel64/libmkl_rt.so 150 && \
    update-alternatives --install /usr/lib/x86_64-linux-gnu/liblapack.so   liblapack.so-x86_64-linux-gnu    /opt/intel/mkl/lib/intel64/libmkl_rt.so 150 && \
    update-alternatives --install /usr/lib/x86_64-linux-gnu/liblapack.so.3 liblapack.so.3-x86_64-linux-gnu  /opt/intel/mkl/lib/intel64/libmkl_rt.so 150 && \
    echo "/opt/intel/lib/intel64"     >  /etc/ld.so.conf.d/mkl.conf && \
    echo "/opt/intel/mkl/lib/intel64" >> /etc/ld.so.conf.d/mkl.conf && \
    ldconfig && \
    echo "MKL_THREADING_LAYER=GNU" >> /etc/environment

RUN git clone https://github.com/facebookresearch/faiss.git && \
    cd faiss && \
    cmake -B build . && \
    make -C build -j faiss && \
    make -C build -j swigfaiss && \
    cd build/faiss/python && \
    pip install --user . && \
    pip install awscli

ENV PATH "$PATH:$HOME/.cargo/bin"
