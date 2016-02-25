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
package be.iminds.iot.dianne.api.nn.eval;

import java.util.Map;
import java.util.UUID;

import be.iminds.iot.dianne.api.nn.learn.LearnProgress;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkInstanceDTO;

/**
 * The Evaluator evaluates a neural network instance on a (portion of a) dataset.
 * 
 *  It collects metrics such as the accuracy and the forward time.
 * 
 * @author tverbele
 *
 */
public interface Evaluator {

	/**
	 * @return uuid of this evaluator - same as the frameworkId this evaluator is deployed on
	 */
	UUID getEvaluatorId();
	
	/**
	 * Evaluate a neural network instance on a (portion of a) dataset
	 * @param nni neural network instance to evaluate
	 * @param dataset dataset to evaluate the nni on
	 * @param config configuration
	 * @return Evaluation 
	 * @throws Exception
	 */
	Evaluation eval(String dataset, Map<String, String> config, NeuralNetworkInstanceDTO nni) throws Exception;

	/**
	 * @return the current progress of the Evaluator
	 */
	EvaluationProgress getProgress();
	
	/**
	 * @return whether or not this learner is busy
	 */
	boolean isBusy();
}
