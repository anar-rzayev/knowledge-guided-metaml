FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

RUN apt-get update

RUN echo python --version
RUN echo pip --version

# Install dependencies
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/London

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget git uuid-runtime sudo unzip libgomp1 liblapack3 \
    openmpi-bin openmpi-common jq gcc libmpich-dev bzip2 \
    build-essential ca-certificates apt-utils tar curl xz-utils clinfo

ENV APP_FOLDER=/app
ENV PYTHONPATH=$APP_FOLDER:$PYTHONPATH
ENV AM_I_IN_A_DOCKER_CONTAINER Yes
# ENV NEPTUNE_API_TOKEN=$NEPTUNE_API_TOKEN  # Uncomment if using Neptune

WORKDIR $APP_FOLDER

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY . $APP_FOLDER

RUN pip install --upgrade huggingface-hub