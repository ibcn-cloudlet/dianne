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

import be.iminds.iot.dianne.api.nn.NeuralNetwork;

/**
 * The GradientProcessor processes the accumulated gradients of a neural network 
 * and calcultes actual deltas to update the parameters with. GradientProcessors
 * can be stacked where each GradientProcessor in the pipeline updates the deltas in
 * a specific way.
 * 
 * @author tverbele
 *
 */
public abstract class GradientProcessor {

	private final GradientProcessor decorated;
	
	protected final NeuralNetwork nn;
	
	public GradientProcessor( NeuralNetwork nn){
		this.nn = nn;
		
		this.decorated = null;
	}
	
	public GradientProcessor(GradientProcessor decorated){
		this.nn = decorated.nn;
		
		this.decorated = decorated;
	}

	final public void calculateDelta(long i){
		if(decorated!=null){
			decorated.calculateDelta(i);
		}
		
		updateDelta(i);
	}
	
	/**
	 * Update the deltas for the neural network instance
	 * @param i the iteration the learner is currently in - can be used for decaying parameters etc
	 */
	protected abstract void updateDelta(long i);
	
}
