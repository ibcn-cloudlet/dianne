package be.iminds.iot.dianne.api.rl;

import java.util.Map;

import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkInstanceDTO;

/**
 * An Agent implements a policy to interact with a specific Environment, using
 * any Reinforcement Learning paradigm (e.g. Q-learning).
 * 
 * @author smbohez
 *
 */
public interface Agent {

	/**
	 * Starts an acting session with the given environment using the given
	 * neural network name to instantiate a network to be used by the agent to
	 * select action (e.g. Q-network). Optionally, a pool can be provided to
	 * which the agent will add experience.
	 * 
	 * @param nni the neural network instance to be used by the agent
	 * @param environment name of the environment that the agent has to interact with
	 * @param experiencePool name of the pool in which the agent can add experience
	 * @param config the agent configuration to use
	 * @throws Exception if the environment of pool are unknown or the agent is busy
	 */
	void act(NeuralNetworkInstanceDTO nni, String environment, String experiencePool, Map<String, String> config) throws Exception;

	/**
	 * Will stop the current acting session, if any are running.
	 */
	void stop();

}
