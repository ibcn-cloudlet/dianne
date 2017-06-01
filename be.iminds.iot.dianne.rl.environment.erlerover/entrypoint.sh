#!/bin/bash

source /opt/ros/indigo/setup.bash
source /home/dianne/ros_catkin_ws/devel/setup.bash

/home/dianne/ardupilot/APMrover2/APMrover2.elf -S -I0 --home -35.363261,149.165230,584,353 --model Gazebo --speedup=1 --autotest-dir /home/dianne/ardupilot/Tools/autotest &

sleep 1

/usr/bin/python /usr/local/bin/mavproxy.py --master 127.0.0.1:14550 --out 127.0.0.1:14551 --cmd="map overlay /home/dianne/ros_catkin_ws/src/ardupilot_sitl_gazebo_plugin/ardupilot_sitl_gazebo_plugin/worlds/outdoor_village/map_w250m_h250m.jpg -35.363261 149.165230" &

sleep 1

./gradlew -q -Parguments="-Dgazebo.headless=true" run.runtime.agent.rover
