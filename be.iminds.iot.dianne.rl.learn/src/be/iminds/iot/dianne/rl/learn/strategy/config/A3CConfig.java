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
package be.iminds.iot.dianne.rl.learn.strategy.config;

import be.iminds.iot.dianne.nn.learn.criterion.CriterionFactory.CriterionConfig;
import be.iminds.iot.dianne.nn.learn.processors.ProcessorFactory.ProcessorConfig;

public class A3CConfig {

	/**
	 * Discount factor
	 */
	public float discount = 0.99f;
	
	/**
	 * The gradient optimization method to use
	 *  * SGD - stochastic gradient descent (optionally with (nesterov) momentum and regularization parameters)
	 *  * Adadelta
	 *  * Adagrad
	 *  * RMSprop
	 */
	public ProcessorConfig method = ProcessorConfig.SGD;

	/**
	 * The criterion to use for the value network
	 */
	public CriterionConfig valueCriterion = CriterionConfig.MSE;
	
	/**
	 * Set to false to only train value function from existing policy
	 */
	public boolean updatePolicy = true;
	
	/**
	 * Rate of entropy regularization
	 */
	public float entropy = 0.0f;
	
	/**
	 * Set true to allow to revisit same sequences in the experience pool
	 */
	public boolean reuseSequences = false;
	
}
