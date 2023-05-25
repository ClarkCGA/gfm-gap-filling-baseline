FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

RUN pip install --upgrade pip
RUN apt-get update && apt-get install -y vim tmux && \
    pip install pip-tools

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY gap-filling-baseline /workspace/gap-filling-baseline

COPY data /workspace/gap-filling-baseline/data

EXPOSE 8888
