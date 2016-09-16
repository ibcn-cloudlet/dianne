# DIANNE Kuka environments using ROS and VREP

The Kuka environments will run either on the real hardware or using a simulator. Currently only the VREP simulator is implemented. The robot is interfaced through ROS.

The following instructions explain how to get a working install to run the Kuka environments in simulation with VREP. These are only tested on an Ubuntu machine (14.04). 

## Installing ROS

First install ROS Indigo on your machine. On Ubuntu 14.04 ROS can be installed from prebuilt packages:

```
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
sudo apt-key adv --keyserver hkp://ha.pool.sks-keyservers.net --recv-key 0xB01FA116
sudo apt-get update
sudo apt-get install ros-indigo-desktop-full
sudo rosdep init
rosdep update
echo "source /opt/ros/indigo/setup.bash" >> ~/.bashrc
source ~/.bashrc
sudo apt-get install python-rosinstall
``` 

For more details, look at the [ROS wiki](http://wiki.ros.org/indigo/Installation/Ubuntu)

## Installing VREP

Download VREP from the [coppeliarobotics website](http://www.coppeliarobotics.com/downloads.html). The version we used is 3.3.1.

On Ubuntu the VREP installation is done by extracting the .tar.gz

## Building VREP ROS Interface

Build the VREP ROS Interface as explained in the [VREP documetation](http://www.coppeliarobotics.com/helpFiles/en/rosInterf.htm) 

## Running the DIANNE environment

To run the DIANNE Kuka environment, make sure you have launched VREP with the ROS Interface enabled, or set the 	vrep.dir property in your bndrun launcher where the VREP installation can be found.






