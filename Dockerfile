# docker build -t c4ml-present:1.0 .
# docker compose up
# First stage : setup the system and environment
FROM ubuntu:22.04 AS base

RUN \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        ca-certificates-java \
        curl \
        graphviz \
        openjdk-8-jre-headless \
        python3-distutils \
			  pandoc \	
				libfindbin-libs-perl \
				vim \
				make \
        && \
    rm -rf /var/lib/apt/lists/*

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3 get-pip.py
RUN pip3 install notebook

RUN useradd -ms /bin/bash bootcamp

ENV SCALA_VERSION=2.12.10
ENV ALMOND_VERSION=0.9.1

ENV COURSIER_CACHE=/coursier_cache

ADD . /workdir/
WORKDIR /workdir

ENV JUPYTER_CONFIG_DIR=/jupyter/config
ENV JUPITER_DATA_DIR=/jupyter/data

RUN mkdir -p $JUPYTER_CONFIG_DIR/custom
RUN cp source/custom.js $JUPYTER_CONFIG_DIR/custom/

# Second stage - download Scala requirements and the Scala kernel
FROM base as intermediate-builder

RUN mkdir /coursier_cache

RUN \
    curl -L -o coursier https://git.io/coursier-cli && \
    chmod +x coursier && \
    ./coursier \
        bootstrap \
        -r jitpack \
        sh.almond:scala-kernel_$SCALA_VERSION:$ALMOND_VERSION \
        --sources \
        --default=true \
        -o almond && \
    ./almond --install --global && \
    \rm -rf almond couriser /root/.cache/coursier 

# Last stage
FROM base AS final

# copy the Scala requirements and kernel into the image 
COPY --from=intermediate-builder /coursier_cache/ /coursier_cache/
COPY --from=intermediate-builder /usr/local/share/jupyter/kernels/scala/ /usr/local/share/jupyter/kernels/scala/

RUN cp -r /coursier_cache /home/bootcamp/coursier_cache
ENV COURSIER_CACHE=/home/bootcamp/coursier_cache
RUN pip install jupyterlab-rise
RUN chown -R bootcamp:bootcamp /workdir
RUN chown -R bootcamp:bootcamp /jupyter/config/
RUN chown -R bootcamp:bootcamp /home/bootcamp/coursier_cache
COPY plugin.jupyterlab-settings /jupyter/config/lab/user-settings/jupyterlab-rise/plugin.jupyterlab-settings
RUN pip install chisel4ml brevitas scikit-learn pandas netron

RUN mkdir /c4ml/
RUN chown -R bootcamp:bootcamp /c4ml/
RUN curl -L https://github.com/cs-jsi/chisel4ml/releases/download/0.3.6/chisel4ml.jar -o /c4ml/chisel4ml.jar
RUN apt update && \
		apt install -y git help2man perl python3 make autoconf g++ flex bison ccache && \
		apt install -y libgoogle-perftools-dev numactl perl-doc libfl2 libfl-dev zlib1g zlib1g-dev
RUN git clone https://github.com/verilator/verilator && \
		cd verilator && \
		git checkout v5.034 && \
		autoconf && \
		./configure && \
    make -j 4  && \
		make install 

WORKDIR /workdir
USER bootcamp
EXPOSE 8888
CMD jupyter-lab --no-browser --ip 0.0.0.0 --port 8888 --allow-root
