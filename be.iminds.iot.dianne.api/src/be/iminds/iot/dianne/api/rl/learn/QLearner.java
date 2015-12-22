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
package be.iminds.iot.dianne.api.rl.learn;

import java.util.Map;

import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkInstanceDTO;

/**
 * QLearner interface for reinforcement learning with target neural network 
 * 
 * @author tverbele
 *
 */
public interface QLearner {

	/**
	 * Start learning for a given neural network and dataset, using the given processor.
	 * The learning will start in a background thread and this method immediately returns.
	 * 
	 * @param nni the neural network instance that should be updated
	 * @param targeti the target neural network instance for Q learning
	 * @param dataset the name of the dataset to process for learning
	 * @param config the learner configuration to use
	 */
	void learn(NeuralNetworkInstanceDTO nni, NeuralNetworkInstanceDTO targeti, String dataset, Map<String, String> config) throws Exception;
	
	/**
	 * Stop the current learning session.
	 */
	void stop();
}
