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
package be.iminds.iot.dianne.api.nn.learn;

/**
 * Represents the progress made by a Learner
 * 
 * @author tverbele
 *
 */
public class LearnProgress {

	/** The number of iterations (=number of batches) processed */
	public final long iteration;
	
	/** The current minibatch loss perceived by the Learner */
	public final float minibatchLoss;
	
	public LearnProgress(long iteration, float loss){
		this.iteration = iteration;
		this.minibatchLoss = loss;
	}
	
	@Override
	public String toString(){
		return "[LEARNER] Iteration: "+iteration+" Loss: "+minibatchLoss;
	}
}
