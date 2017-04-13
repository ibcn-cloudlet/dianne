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

public class ValueIterationConfig extends DeepQConfig {

	/**
	 * Episode length to sample
	 */
	public int episodeLength = 100;
	
	/**
	 * Whether to regard the episode as terminal or not
	 */
	public boolean episodeTerminal = true;
	
	/**
	 * Number of steps for N-step Q
	 */
	public int Qsteps = 1;
	
	/**
	 * Number of state dimensions (required as we're not using a pool)
	 */
	public int stateDims = 1;
	
	/**
	 * Number of action dimensions (required as we're not using a pool)
	 */
	public int actionDims = 1;
}
