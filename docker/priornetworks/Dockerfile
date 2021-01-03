# FROM docker.io/deepprojects/cuda-sshfs:9.0-3.5.1
FROM pytorch/pytorch

RUN apt-get update \
&& apt-get install -y sshfs gcc curl\
&& pip install --upgrade pip

# install connectors
RUN pip install ipykernel && python3 -m ipykernel install --user && pip install papermill
RUN pip install red-connector-ssh red-connector-http

# symlink for python3 shebang
RUN ln -s /opt/conda/bin/python3 /usr/bin/python3

# create user
RUN useradd -ms /bin/bash cc
USER cc

# install papermill wrapper for jupyter
RUN mkdir -p "/home/cc/.local/bin" \
	&& curl https://raw.githubusercontent.com/curious-containers/cc-jupyter-service/master/cc_jupyter_service/papermill_wrapper.py > /home/cc/.local/bin/papermill_wrapper.py \
	&& chmod u+x /home/cc/.local/bin/papermill_wrapper.py

# add content for curious containers
COPY --chown=cc:cc requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY --chown=cc:cc src /home/cc/src
RUN mkdir /home/cc/resources
COPY --chown=cc:cc resources/data /home/cc/resources/data
RUN ln -s /home/cc/src/main_train_new_network.py /home/cc/.local/bin/train_new_network

# set environment
ENV PATH="/home/cc/.local/bin:${PATH}"
ENV LC_ALL="C.UTF-8"
ENV LANG="C.UTF-8"
ENV CC_HOME="/home/cc"
WORKDIR /home/cc
