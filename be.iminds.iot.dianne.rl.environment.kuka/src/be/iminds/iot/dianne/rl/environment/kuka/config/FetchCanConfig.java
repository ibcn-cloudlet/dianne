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
	 * margin used to determine success in case of early stopping
	 */
	public float margin = 0.005f;
	
	/**
	 * environment difficulty (the higher the more difficult)
	 * 
	 * can be used for curriculum learning
	 * 
	 * difficulty = -1 : fixed youBot, fixed Can
	 * difficulty = 0 : fixed youBot, Can right in front
	 * difficulty = 1 : fixed youBot, Can in front in sight
	 * difficulty = 2 : random youBot / Can positions
	 * ...
	 * 
	 * TODO make enum for this?
	 */
	public int difficulty = 2;
	
	/**
	 * Give intermediate reward on each action based on the distance towards Can
	 */
	public boolean intermediateReward = true;
	
	/**
	 * In case of intermediateReward, give reward on relative distance covered towards Can 
	 * (in case of false return the (normalized) distance to Can of the state)
	 */
	public boolean relativeReward = false;
	
	/**
	 * In case of relativeReward, use this scale factor to multiply diff in distance to the Can with
	 */
	public float relativeRewardScale = 5.0f;
	
	/**
	 * In case of intermediateReward, give reward based on exponential decaying function.
	 * (in case of false use the linear reward function)
	 */
	public boolean exponentialDecayingReward = false;
	
	/**
	 * In case of exponentialDecayingReward, use this scale factor to modify the decay rate.
	 * (]0,R[)
	 */
	public float exponentialDecayingRewardScale = 2.5f;
	
	/**
	 * Only give +1 or -1 rewards in case of relative rewards
	 */
	public boolean discreteReward = false;
	
	/**
	 * Punish wrong grip action with a -1 reward
	 */
	public boolean punishWrongGrip = false;
	
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
	
	/**
	 * Whether the environment has to tick the simulator
	 * 
	 * Set to false when using simulator to mimick real world ...
	 */
	public boolean tick = true;
	
	/**
	 * Number of sensors to activate in the environment
	 */
	public int environmentSensors = 0;
	
	/**
	 * Ms to wait at initialization time before regarding this as failure
	 */
	public int timeout = 100000;
	
	/**
	 * A seed for initializing the environment
	 */
	public long seed = 0;

	/**
	 * Whether a collision in the environment is terminal.
	 * 
	 * Set to false if the reward never gets positive.
	 */
	public boolean collisionTerminal = false;

	/**
	 * Scale factor to modify the reward of a grip action.
	 */
	public float gripRewardScale = 1.0f;

	/**
	 * Offset for the reward function. Can be used to make a reward function positive or negative.
	 */
	public float maxReward = 0.0f;
}
