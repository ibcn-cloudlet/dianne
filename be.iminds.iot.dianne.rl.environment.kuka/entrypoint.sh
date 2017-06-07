#!/bin/bash

source /opt/ros/indigo/setup.bash
source /opt/ros_catkin_ws/devel/setup.bash

Xvfb :99 &

cd /home/dianne/tools
java -Dvrep.headless=true $* -jar generated/distributions/executable/runtime.agent.kuka.jar
