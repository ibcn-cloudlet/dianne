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
package be.iminds.iot.dianne.rl.environment.erlerover.config;

public class ErleroverConfig {
	
	/**
	 * Number of discrete actions to allow 3-6
	 */
	public int numActions = 6;

	/**
	 * Min laser sensor value below which we determine collision
	 */
	public float crashThreshold = 0.4f;
	
	/**
	 * Add penalty to reward the closer you are to a wall : negative scale factor for (1 - shortest normalized distance measured to wall)
	 */
	public float dangerousPenalty = 0.0f;
	
	/**
	 * Add penalty to reward for steering : negative scale factor for abs(yaw)
	 */
	public float steeringPenalty = -0.5f;
	
	/**
	 * Penalty for crashing
	 */
	public float crashPenalty = -1.0f;
	
	/**
	 * Circuits to use in the agent
	 */
	public String[] circuits = new String[]{"circuit1","circuit2","circuit3"};
	
}
