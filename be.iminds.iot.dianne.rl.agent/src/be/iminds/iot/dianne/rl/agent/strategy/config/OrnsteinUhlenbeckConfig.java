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
 *     Steven Bohez
 *******************************************************************************/
package be.iminds.iot.dianne.rl.agent.strategy.config;

public class OrnsteinUhlenbeckConfig {

	/**
	 * Max noise
	 */
	public double noiseMax = 1e0;
	
	/**
	 * Min noise
	 */
	public double noiseMin = 0;
	
	/**
	 * Noise exponential decay rate
	 */
	public double noiseDecay = 1e-6;
	
	/**
	 * Minimum value
	 */
	public float minValue = Float.NEGATIVE_INFINITY;
	
	/**
	 * Maximum value
	 */
	public float maxValue = Float.POSITIVE_INFINITY;

	/**
	 * 
	 */
	public float[] theta = {0.15f};
	
	/**
	 * 
	 */
	public float[] mu = {0.00f};
	
	/**
	 * 
	 */
	public float[] sigma = {0.30f};
}
