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
package be.iminds.iot.dianne.nn.learn.processors;

import be.iminds.iot.dianne.api.log.DataLogger;
import be.iminds.iot.dianne.api.nn.NeuralNetwork;
import be.iminds.iot.dianne.api.nn.learn.Processor;
import be.iminds.iot.dianne.tensor.TensorFactory;

public abstract class AbstractProcessor implements Processor {

	private final AbstractProcessor decorated;
	
	protected final TensorFactory factory;
	protected final NeuralNetwork nn;
	protected final DataLogger logger;
	
	public AbstractProcessor(TensorFactory factory, 
			NeuralNetwork nn,
			DataLogger logger){
		this.factory = factory;
		this.nn = nn;
		this.logger = logger;
		
		this.decorated = null;
	}
	
	public AbstractProcessor(AbstractProcessor decorated){
		this.factory = decorated.factory;
		this.nn = decorated.nn;
		this.logger = decorated.logger;
		
		this.decorated = decorated;
	}

	@Override
	final public float processNext(){
		float error = 0;
		
		if(decorated!=null){
			error = decorated.processNext();
		}
		
		error = processNext(error);
		return error;
	}
	
	protected abstract float processNext(float error);
	
}
