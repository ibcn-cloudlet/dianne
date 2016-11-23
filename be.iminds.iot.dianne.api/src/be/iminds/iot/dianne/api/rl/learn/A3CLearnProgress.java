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
package be.iminds.iot.dianne.api.rl.learn;

import be.iminds.iot.dianne.api.nn.learn.LearnProgress;
import be.iminds.iot.dianne.tensor.Tensor;

/**
 * Represents the progress made by a Q Learner
 * 
 * @author tverbele
 *
 */
public class A3CLearnProgress extends LearnProgress{

	/** The average state-value */
	public final float value;
	public final float entropy;
	
	public A3CLearnProgress(long iteration, float loss, float v, float e){
		super(iteration, loss);
		this.value = v;
		this.entropy = e;
	}
	
	@Override
	public String toString(){
		return "[LEARNER] Iteration: "+iteration+" Loss: "+minibatchLoss+" V: "+value+" Entropy: "+entropy;
	}
}
