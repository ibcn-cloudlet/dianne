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
package be.iminds.iot.dianne.api.nn.learn;

import java.util.Map;
import java.util.UUID;

import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkInstanceDTO;

/**
 * The Learner implements the flow of the learning process. The Learner gets an instance of 
 * the neural network to learn, instantiates the Processor to process items from the Dataset, 
 * and pushes weight updates to the DianneRepository.
 * 
 * @author tverbele
 *
 */
public interface Learner {

	/**
	 * @return uuid of this learner - same as the frameworkId this learner is deployed on
	 */
	UUID getLearnerId();
	
	/**
	 * Start learning for a given neural network and dataset, using the given processor.
	 * The learning will start in a background thread and this method immediately returns.
	 * 
	 * @param dataset the name of the dataset to process for learning
	 * @param config the learner configuration to use
	 * @param nni the neural network instance(s) that should be used for learning
	 *
	 */
	void learn(String dataset, Map<String, String> config, NeuralNetworkInstanceDTO... nni) throws Exception;
	
	/**
	 * @return the current progress of the Learner
	 */
	LearnProgress getProgress();
	
	/**
	 * @return whether or not this learner is busy
	 */
	boolean isBusy();
	
	/**
	 * Stop the current learning session.
	 */
	void stop();
}
