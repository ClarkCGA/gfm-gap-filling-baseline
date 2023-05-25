FROM nvidia/cuda:12.1.1-devel-ubi8

# Install Miniconda
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV PATH=/opt/conda/bin:$PATH

RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://repo.anaconda.com/miniconda/Miniconda3-py39_4.10.3-Linux-x86_64.sh -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -bfp /opt/conda \
    && rm -rf /tmp/miniconda.sh \
    && conda clean -tipsy \
    && find /opt/conda/ -follow -type f -name '*.a' -delete \
    && find /opt/conda/ -follow -type f -name '*.js.map' -delete \
    && /opt/conda/bin/conda clean -afy

RUN pip install --upgrade pip
RUN apt-get update && apt-get install -y vim tmux && \
    pip install pip-tools

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN mkdir /home/workdir
WORKDIR /home/workdir

EXPOSE 8888
