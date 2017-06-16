FROM dianne

# install Gym dependencies
USER root

RUN apt-get update &&\
    apt-get install -y \
 	python-dev \
	python-pygame \
	python-setuptools \
	swig \
	xvfb \
	&&\
    apt-get clean &&\
    rm -rf /var/lib/apt/lists/* &&\
    easy_install pip

# install Gym
WORKDIR /opt

RUN git clone https://github.com/openai/gym &&\
    cd gym &&\
    pip install -e .

# install box2d
RUN git clone https://github.com/pybox2d/pybox2d && \
	cd pybox2d/ && \
	python setup.py build && \
	python setup.py install

# install Jep
RUN pip install jep
ENV LD_PRELOAD /usr/lib/x86_64-linux-gnu/libpython2.7.so


# go back to DIANNE home
WORKDIR ${home}
USER dianne


# build and export Gym runtime
RUN ./gradlew -x :be.iminds.iot.dianne.rl.environment.ale:assemble assemble export.runtime.agent.gym

# set default target
ENV TARGET runtime.agent.gym

# run
ENTRYPOINT ["/home/dianne/be.iminds.iot.dianne.rl.environment.gym/entrypoint.sh"]
