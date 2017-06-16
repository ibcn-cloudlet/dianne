#!/bin/bash

Xvfb :99 &

cd /home/dianne/tools
java $* -jar generated/distributions/executable/$TARGET.jar
