FROM dianne

# install ALE dependencies
USER root

RUN apt-get update && \ 
    apt-get install -y \ 
	libsdl1.2-dev \
	libsdl-gfx1.2-dev \
	libsdl-image1.2-dev

USER dianne


# build and export ALE runtime
RUN ./gradlew assemble export.runtime.agent.ale

# set default target
ENV TARGET runtime.agent.ale

# run
ENTRYPOINT ["/home/dianne/be.iminds.iot.dianne.rl.environment.ale/entrypoint.sh"]
