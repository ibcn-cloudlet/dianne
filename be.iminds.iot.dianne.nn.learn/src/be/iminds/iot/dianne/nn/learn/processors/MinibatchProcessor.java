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

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.log.DataLogger;
import be.iminds.iot.dianne.api.nn.NeuralNetwork;
import be.iminds.iot.dianne.api.nn.learn.Criterion;
import be.iminds.iot.dianne.api.nn.learn.SamplingStrategy;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;

public class MinibatchProcessor extends AbstractProcessor {

	// dataset
	protected final Dataset dataset;
	// sample strategy
	protected final SamplingStrategy sampling;
	// error criterion
	protected final Criterion criterion;
	// batch size
	protected final int batchSize;
	
	// current error
	protected float error = 0;
	
	public MinibatchProcessor(TensorFactory factory, 
			NeuralNetwork nn, 
			DataLogger logger, 
			Dataset d, 
			SamplingStrategy s,
			Criterion c,
			int batchSize) {
		super(factory, nn, logger);
	
		this.dataset = d;
		this.sampling = s;
		this.criterion = c;
		
		this.batchSize = batchSize;
	}
	
	
	@Override
	public float processNext(float e) {
		error = 0;

		for(int i=0;i<batchSize;i++){
			// new sample
			int index = sampling.next();
			Tensor in = dataset.getInputSample(index);

			// forward
			Tensor out = nn.forward(in, ""+index);
			
			// evaluate criterion
			Tensor gradOut = getGradOut(out, index);
			
			// backward
			Tensor gradIn = nn.backward(gradOut, ""+index);
			
			// acc grad params
			accGradParameters();
		}
		
		return error/batchSize;
	}

	protected void accGradParameters(){
		// acc gradParameters
		nn.getTrainables().values().stream().forEach(m -> m.accGradParameters());
	}
	
	protected Tensor getGradOut(Tensor out, int index){
		Tensor e = criterion.error(out, dataset.getOutputSample(index));
		error += e.get(0);
		return criterion.grad(out, dataset.getOutputSample(index));
	}
}
