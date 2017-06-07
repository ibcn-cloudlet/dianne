#!/bin/bash

Xvfb :99 &

cd /home/dianne/tools
java $* -jar generated/distributions/executable/runtime.agent.gym.jar
