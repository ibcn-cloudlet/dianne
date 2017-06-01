#!/bin/bash

xvfb-run -s "-screen 0 1400x900x24" ./gradlew -q run.runtime.agent.gym
