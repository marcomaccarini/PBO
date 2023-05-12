FROM continuumio/miniconda3
COPY . /PBO_rest
WORKDIR /PBO_rest
RUN apt-get -y update
ENV FLASK_APP=main/index.py
ENV FLASK_RUN_HOST=127.0.0.1
RUN conda env create -f env.yml
ENV PATH /opt/conda/envs/pbo_rest/bin:$PATH
RUN /bin/bash -c "source activate pbo_rest"
EXPOSE 5000