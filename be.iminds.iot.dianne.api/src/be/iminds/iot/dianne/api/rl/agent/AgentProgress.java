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
package be.iminds.iot.dianne.api.rl.agent;

public class AgentProgress {
	
	/**
	/* every time the agent gets a parameter update, episode is increased by 1
	 */
	public long episode;
	
	/**
	 * the sequence the agent executed in this act job
	 */
	public long sequence;
	
	/**
	 * number of iterations executed in this sequence
	 */
	public long iterations;
	
	/**
	 * total reward gathered in the sequence
	 */
	public float reward;
	
	public AgentProgress(long s, long i, float r, long e){
		this.sequence = s;
		this.iterations = i;
		this.reward = r;
		this.episode = e;
	}
	
	@Override
	public String toString(){
		return "[AGENT] Sequence: "+sequence+" Iterations: "+iterations+" Reward: "+reward+" Episode: "+episode;
	}
}
