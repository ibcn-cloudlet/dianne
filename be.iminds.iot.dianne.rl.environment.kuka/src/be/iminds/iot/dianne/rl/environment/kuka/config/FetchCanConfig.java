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
package be.iminds.iot.dianne.rl.environment.kuka.config;

public class FetchCanConfig {

	/**
	 * stop early when simulating
	 * 
	 * this will not simulate the grip action, and calculate end reward purely based on end position
	 */
	public boolean earlyStop = true;
	
	/**
	 * environment difficulty (the higher the more difficult)
	 * 
	 * can be used for curriculum learning
	 * 
	 * difficulty = 0 : fixed youBot, Can right in front
	 * difficulty = 1 : fixed youBot, Can in front in sight
	 * difficulty = 2 : random youBot / Can positions
	 * ...
	 * 
	 * TODO make enum for this?
	 */
	public int difficulty = 2;
	
	/**
	 * Give intermediate reward on each action based on the distance covered towards Can
	 */
	public boolean intermediateReward = true;
	
	/**
	 * Skip a number of frames after each action, keeping executing the same action
	 */
	public int skip = 0;
	
	/**
	 * Speed (in m/s) the Kuka base is steered (max speed in case of continuous actions)
	 */
	public float speed = 0.1f;
	
	/**
	 * Max number of actions before terminating anyhow
	 */
	public int maxActions = 100;
}
