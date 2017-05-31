#!/bin/bash

source /opt/ros/indigo/setup.bash
source /home/dianne/ros_catkin_ws/devel/setup.bash

Xvfb :99 &
./gradlew -q -Parguments="-Dvrep.headless=true" run.runtime.agent.kuka
