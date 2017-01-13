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
package be.iminds.iot.dianne.rl.environment.gym.config;


public class GymConfig {

	/**
	 * Environment to load
	 */
	public String env = "CartPole-v0";

	/**
	 * Render the Gym environment on the machine
	 */
	public boolean render = true;
	
	/**
	 * Reward factor
	 */
	public float rewardFactor = 0.01f;

	/**
	 * action factor. Change range of the action.
	 */
	public float actionFactor = 1.0f;
	
	/**
	 * Max number of actions before terminating anyhow
	 */
	public int maxActions = -1;
}
