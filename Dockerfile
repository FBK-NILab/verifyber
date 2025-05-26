FROM continuumio/miniconda3:latest

ENV CONDA_ENV_NAME=verifyber

WORKDIR /app 

RUN conda create -y -n $CONDA_ENV_NAME python=3.8 \
    && conda install -y -n $CONDA_ENV_NAME pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.3 -c pytorch \
    && conda install -y -n $CONDA_ENV_NAME pyg=*=*cu* -c pyg \
    && conda install -y -n $CONDA_ENV_NAME mrtrix3=3.0 -c mrtrix3 \
    && conda clean -afy

# Activate the environment by default
SHELL ["conda", "run", "-n", "verifyber", "/bin/bash", "-c"]

# Additional dependencies
RUN pip install antspyx==0.4.2 dipy==1.7.0 torchviz numba tensorboard
RUN conda install pytorch-cluster -c pyg

# Copy the project files
COPY . .

# Setup the configuration file. This is meant to be
# overridden by the user at runtime, by mounting its
# configuration file at runtime.
ARG CONFIG_PATH=/app/run_config_file.json
COPY run_config.json ${CONFIG_PATH}

# User can simply mount this path to the appropriate configuration file on the host machine.
ENV VERIFYBER_DEFAULT_CONFIG=${CONFIG_PATH}
ENV VERIFYBER_OUTPUT_DIR=/app/output

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "verifyber", "python", "tractogram_filtering.py"]
