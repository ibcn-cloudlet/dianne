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

	public enum Difficulty {
		FIXED,
		WORKSPACE,
		VISIBLE,
		RANDOM,
		START_DOCKED,
		RANDOM_DOCK
	}
	
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
	 * difficulty = 0  (FIXED): fixed youBot, fixed Can
	 * difficulty = 1  (WORKSPACE): fixed youBot, Can right in front
	 * difficulty = 2  (VISIBLE): fixed youBot, Can in front in sight
	 * difficulty = 3  (RANDOM): random youBot / Can positions
	 * difficulty = 4  (START_DOCKED): start youBot in docked position
	 * difficulty = 5  (RANDOM_DOCK): start youBot randomly with also dock present
	 */
	public Difficulty difficulty = Difficulty.RANDOM;
		
	/**
	 * Distance scale factor used to spread or tighten reward functions.
	 */
	public float distanceScale = 1.0f;
		
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
	public float speed = 0.2f;

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
	 * Use relative distance from the gripper instead of from the base platform
	 */
	public boolean gripperDistance = false;
	
	/**
	 * Offset for the reward function. Can be used to make a reward function positive or negative.
	 */
	public float rewardOffset = 0.0f;

	/**
	 * Scale factor to modify the reward.
	 * 
	 * A scale factor lower than one reduces the reward and a factor higher than one boosts the reward.
	 */
	public float rewardScale = 1.0f;
	
	public enum IntermediateReward {
		NONE,
		RELATIVE_DISCRETE,
		RELATIVE_CONTINUOUS,
		LINEAR,
		EXPONENTIAL,
	}
	
	/**
	 * Give intermediate reward on each action based on the distance towards the Can.
	 * 
	 * intermediateReward = 0  (NONE): no intermediate rewards
	 * intermediateReward = 1  (RELATIVE_DISCRETE): discrete reward based on relative distance covered towards Can: 0,1,-1
	 * intermediateReward = 2  (RELATIVE_CONTINUOUS): reward based on relative distance covered towards Can
	 * intermediateReward = 3  (LINEAR): reward based on negative linear distance
	 * intermediateReward = 4  (EXPONENTIAL): reward based on exponential decaying function (sharp edge at 0)
	 * 
	 * default = LINEAR
	 */
	public IntermediateReward intermediateReward = IntermediateReward.LINEAR;
	
}
