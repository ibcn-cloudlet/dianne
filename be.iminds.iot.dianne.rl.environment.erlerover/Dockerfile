FROM dianne

USER root

# install base packages and MAVProxy dependencies
RUN apt-get update && \
    apt-get -y upgrade && \
    apt-get -y -f install g++ gawk python-pip python-matplotlib python-serial python-wxgtk2.8 python-scipy python-opencv python-numpy python-pyparsing ccache realpath libopencv-dev


# use bash instead of sh
RUN rm /bin/sh && ln -s /bin/bash /bin/sh


# install MAVProxy
RUN pip install future && \
    apt-get install libxml2-dev libxslt1-dev -y && \
    pip2 install pymavlink catkin_pkg --upgrade && \
    pip install MAVProxy==1.5.2


# install ROS Indigo
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' && \ 
    apt-key adv --keyserver hkp://ha.pool.sks-keyservers.net --recv-key 0xB01FA116 && \
    apt-get update && \
    apt-get install -y ros-indigo-ros-base \
			python-rosinstall \
                        ros-indigo-octomap-msgs    \
                        ros-indigo-joy             \
                        ros-indigo-geodesy         \
                        ros-indigo-octomap-ros     \
                        ros-indigo-mavlink         \
                        ros-indigo-control-toolbox \
                        ros-indigo-transmission-interface \
                        ros-indigo-joint-limits-interface \
                        unzip && \
    rosdep init


# source ROS stuff
RUN echo "source /opt/ros/indigo/setup.bash" >> /home/dianne/.bashrc &&\
    echo "source /opt/ros_catkin_ws/devel/setup.bash" >> /home/dianne/.bashrc

# install Gazebo
RUN sh -c 'echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable `lsb_release -cs` main" > /etc/apt/sources.list.d/gazebo-stable.list' &&\
    wget http://packages.osrfoundation.org/gazebo.key -O - | apt-key add - && \
    apt-get update && \
    apt-get remove .*gazebo.* '.*sdformat.*' '.*ignition-math.*' && apt-get update && apt-get install -y gazebo7 libgazebo7-dev drcsim7 xvfb


# download Gazebo models
RUN mkdir -p ~/.gazebo/models && \
    git clone https://github.com/erlerobot/erle_gazebo_models && \
    mv erle_gazebo_models/* ~/.gazebo/models

WORKDIR /opt

RUN mkdir -p ros_catkin_ws/src &&\
    chown -R dianne:dianne ros_catkin_ws &&\
    mkdir ardupilot &&\
    chown dianne:dianne ardupilot

USER dianne

# download additional ROS packages
RUN rosdep update && \
    git clone https://github.com/erlerobot/ardupilot -b gazebo && \
    source /opt/ros/indigo/setup.bash && \
    echo "source /opt/ros/indigo/setup.bash" >> ~/.bashrc && \
    cd ros_catkin_ws/src && \
    catkin_init_workspace && \
    cd .. && \
    catkin_make && \
    echo "source /opt/ros_catkin_ws/devel/setup.bash" >> ~/.bashrc && \
    cd src && \
    git clone https://github.com/erlerobot/ardupilot_sitl_gazebo_plugin && \
    git clone https://github.com/tu-darmstadt-ros-pkg/hector_gazebo/ && \
    git clone https://github.com/erlerobot/rotors_simulator -b sonar_plugin && \
    git clone https://github.com/PX4/mav_comm.git && \
    git clone https://github.com/ethz-asl/glog_catkin.git && \
    git clone https://github.com/catkin/catkin_simple.git && \
    git clone https://github.com/erlerobot/mavros.git && \
    git clone https://github.com/ros-simulation/gazebo_ros_pkgs.git -b indigo-devel


# patch and build ROS packages from source
RUN cd ros_catkin_ws && \
    source devel/setup.bash && \
    cd src/ardupilot_sitl_gazebo_plugin && \
    git apply /home/dianne/be.iminds.iot.dianne.rl.environment.erlerover/ardupilot_sitl_gazebo_plugin.patch && \
    cd ../.. && \
    catkin_make --pkg mav_msgs mavros_msgs gazebo_msgs && \
    source devel/setup.bash && \
    catkin_make -j 4

# build APMrover
RUN cd ardupilot/APMrover2 && make sitl -j 4

# Go back to dianne home
WORKDIR ${home}
USER dianne

# build and export Rover runtime
RUN ./gradlew -x :be.iminds.iot.dianne.rl.environment.ale:assemble assemble export.runtime.agent.rover

ENV DISPLAY :99

# set default target
ENV TARGET runtime.agent.rover

# run
ENTRYPOINT ["/home/dianne/be.iminds.iot.dianne.rl.environment.erlerover/entrypoint.sh"]
