# DIANNE OpenAI Gym environments

The DIANNE OpenAI Gym environments are enabled through Java Embedded Python (JEP), a native interface enabling to call Python code from Java through CPython.

In order for the environments to work, one needs to have installed:

* Python (2.7)
* OpenAI Gym
* Additional simulators (e.g. Box2D or Mujoco)

## Install Jep

Jep embeds CPython in Java through JNI and is used to communicate with the Python based OpenAI gym. Install Jep using pip:

```
pip install jep
```

## Installing OpenAI gym

Fetch the gym source and install:

```
git clone https://github.com/openai/gym
cd gym
pip install -e . # minimal install
```

To perform a full install:

```
pip install -e .[all]
```

More info on OpenAI gym, take a look at their [website](https://gym.openai.com/docs)

## Running an OpenAI environment in DIANNE

To run an OpenAI environment in DIANNE, select "Gym" as DIANNE environment id and pass as config "env=CartPole-v0" to select the actual gym environment to launch.



 