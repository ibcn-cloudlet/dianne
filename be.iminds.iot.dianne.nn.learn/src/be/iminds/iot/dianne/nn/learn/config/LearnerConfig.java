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
package be.iminds.iot.dianne.nn.learn.config;

public class LearnerConfig {

	/**
	 * The tag under which to publish the trained parameters
	 */
	public String tag;
	
	/**
	 * Start the training with new randomized parameters
	 */
	public boolean clean = false;
	
	/**
	 * Output intermediate results to the console
	 */
	public boolean trace = false;
	
	/**
	 * Retry NaNretry times with last sync-ed parameters in case of a NaN error
	 */
	public int NaNretry = 0;
	
	/**
	 * Sync delta parameters with repository each syncInterval batches
	 */
	public int syncInterval = 1000;
	
	public enum Criterion {
		MSE,
		NLL,
		ABS
	}
	
	/**
	 * The criterion to use to evaluate the error between output and target
	 */
	public Criterion criterion = Criterion.MSE;
	
	public enum Method {
		SGD,
		ADADELTA,
		ADAGRAD,
		RMSPROP,
		ADAM
	}
	
	/**
	 * The gradient optimization method to use
	 *  * SGD - stochastic gradient descent (optionally with (nesterov) momentum and regularization parameters)
	 *  * Adadelta
	 *  * Adagrad
	 *  * RMSprop
	 */
	public Method method = Method.SGD;
	
	public enum Sampling {
		RANDOM,
		SEQUENTIAL
	}
	
	/**
	 * The sampling strategy to use to traverse the dataset
	 *  * Random
	 *  * Sequential
	 */
	public Sampling sampling = Sampling.RANDOM;
	
}
