/*******************************************************************************
 * DIANNE  - Framework for distributed artificial neural networks
 * Copyright (C) 2015  iMinds - IBCN - UGent
 *
 * This file is part of DIANNE.
 *
 * DIANNE is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * Contributors:
 *     Tim Verbelen, Steven Bohez
 *******************************************************************************/
package be.iminds.iot.dianne.api.rl.agent;

import java.util.Map;
import java.util.UUID;

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
	 * @return uuid of this agent - same as the frameworkId this agent is deployed on
	 */
	UUID getAgentId();
	
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
	void act(String experiencePool, Map<String, String> config, NeuralNetworkInstanceDTO nni, String environment) throws Exception;

	/**
	 * Will stop the current acting session, if any are running.
	 */
	void stop();

	/**
	 * @return whether or not this learner is busy
	 */
	boolean isBusy();
}
