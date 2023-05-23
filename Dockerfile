FROM continuumio/miniconda3 AS compile-image
COPY ./env.yml ./env.yml
RUN apt-get -y update
RUN conda env create -f env.yml

FROM continuumio/miniconda3 AS build-image
COPY --from=compile-image /opt/conda/envs/pbo_rest /opt/conda/envs/pbo_rest
COPY . /PBO_rest
WORKDIR /PBO_rest

ENV FLASK_APP=main/index.py
ENV FLASK_RUN_HOST=127.0.0.1
ENV PATH /opt/conda/envs/pbo_rest/bin:$PATH
RUN /bin/bash -c "source activate pbo_rest"
EXPOSE 5000