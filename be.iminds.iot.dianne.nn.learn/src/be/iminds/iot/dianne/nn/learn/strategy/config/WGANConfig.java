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
package be.iminds.iot.dianne.nn.learn.strategy.config;

import be.iminds.iot.dianne.nn.learn.processors.ProcessorFactory.ProcessorConfig;
import be.iminds.iot.dianne.nn.learn.sampling.SamplingFactory.SamplingConfig;

public class WGANConfig {

	/**
	 * Batch size in which samples are processed before updating parameters
	 */
	public int batchSize = 1;
	
	/**
	 * The gradient optimization method to use
	 */
	public ProcessorConfig method = ProcessorConfig.RMSPROP;
	
	/**
	 * The learningRate to use
	 */
	public float learningRate = 1e-4f;
	
	/**
	 * The sampling strategy to use to sample real data
	 *  * Random
	 *  * Sequential
	 */
	public SamplingConfig sampling = SamplingConfig.UNIFORM;

	/**
	 * Input dimension of the generator (TODO get this from the NN?)
	 */
	public int generatorDim;
	
	/**
	 * Values to clamp the discriminator weight
	 */
	public float clamp = 0.01f;
	
	/**
	 * Number of iterations to train D before G
	 */
	public int Diterations = 5;
	
	/**
	 * Number of iterations to train D the first initIterations
	 */
	public int initDiterations = 100;
	
	/**
	 * Number of first iterations to use initDiterations
	 */
	public int initIterations = 25;
}
