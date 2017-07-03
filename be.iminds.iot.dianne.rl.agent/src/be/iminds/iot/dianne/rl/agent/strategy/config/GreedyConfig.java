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
package be.iminds.iot.dianne.rl.agent.strategy.config;


public class GreedyConfig {

	/**
	 * Maximal epsilon 
	 */
	public double epsilonMax = 1e0;
	
	/**
	 * Minimal epsilon
	 */
	public double epsilonMin = 0;
	
	/**
	 * Exponential decay rate to decay epsilon from epsilonMax to epsilonMin
	 */
	public double epsilonDecay = 1e-6;

	/**
	 * Momentum : the probabiblity that the previous action is repeated in case of a "random" action
	 */
	public float momentum = 0.0f;
	
	/**
	 * Trace the current epsilon value
	 */
	public boolean trace = false;
	
	/**
	 * The interval to trace
	 */
	public int traceInterval=1000;
}
