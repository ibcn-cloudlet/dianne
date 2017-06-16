FROM dianne

USER root

# install ROS
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' &&\
	apt-key adv --keyserver hkp://ha.pool.sks-keyservers.net --recv-key 0xB01FA116 &&\
	apt-get update &&\
	apt-get install -y ros-indigo-desktop-full python-rosinstall python-catkin-tools wget curl &&\
	rosdep init

# use bash instead of sh
RUN rm /bin/sh && ln -s /bin/bash /bin/sh

WORKDIR /opt

RUN mkdir -p ros_catkin_ws &&\
    chown dianne:dianne ros_catkin_ws

USER dianne

RUN rosdep update &&\
    source /opt/ros/indigo/setup.bash &&\
    cd ros_catkin_ws &&\
    git clone https://github.com/ibcn-cloudlet/ros.git src &&\
    cd src &&\
    git submodule init &&\
    git submodule update &&\
    catkin_init_workspace &&\
    cd .. &&\
    catkin build --continue-on-failure; exit 0

USER root

# source ROS stuff
RUN echo "source /opt/ros/indigo/setup.bash" >> /home/dianne/.bashrc &&\
    echo "source /opt/ros_catkin_ws/devel/setup.bash" >> /home/dianne/.bashrc

# install V-REP 
RUN curl http://coppeliarobotics.com/files/V-REP_PRO_EDU_V3_4_0_Linux.tar.gz | tar -xz &&\
    mv V-REP_PRO_EDU_V3_4_0_Linux /opt/vrep &&\
    cp ros_catkin_ws/devel/lib/libv_repExtRosInterface.so /opt/vrep/

# V-REP depends on qt5 and needs (virtual) display (yes, also in headless mode...)
USER root
RUN apt-get update &&\
    apt-get install -y xvfb qt5-default

# Go back to dianne home
WORKDIR ${home}
USER dianne

# build and export Kuka runtime
RUN ./gradlew -x :be.iminds.iot.dianne.rl.environment.ale:assemble assemble export.runtime.agent.kuka

ENV DISPLAY :99

# set default target
ENV TARGET runtime.agent.kuka

# run
ENTRYPOINT ["/home/dianne/be.iminds.iot.dianne.rl.environment.kuka/entrypoint.sh"]
