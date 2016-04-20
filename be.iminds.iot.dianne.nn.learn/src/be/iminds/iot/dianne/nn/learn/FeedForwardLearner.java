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
package be.iminds.iot.dianne.nn.learn;

import java.util.Map;

import org.osgi.service.component.annotations.Component;

import be.iminds.iot.dianne.api.dataset.Sample;
import be.iminds.iot.dianne.api.nn.learn.Learner;
import be.iminds.iot.dianne.api.nn.module.Trainable;
import be.iminds.iot.dianne.tensor.Tensor;

@Component(service=Learner.class, 
	property={"aiolos.unique=true",
			  "dianne.learner.category=FF"})
public class FeedForwardLearner extends AbstractLearner {
	
	protected int batchSize = 10;
	protected boolean batchAverage = true;

	protected void loadConfig(Map<String, String> config){
		super.loadConfig(config);
		
		if(config.containsKey("batchSize"))
			batchSize = Integer.parseInt(config.get("batchSize"));
		
		if(config.containsKey("batchAverage"))
			batchAverage = Boolean.parseBoolean(config.get("batchAverage"));
		
		System.out.println("* batchSize = " +batchSize);
		System.out.println("* batchAverage = " + batchAverage);
	}
	
	protected float process(long i){
		// Clear delta params
		nn.getTrainables().values().stream().forEach(Trainable::zeroDeltaParameters);
		
		// Process batch
		float err = 0;
		for(int k=0; k<batchSize; k++){
			// Select new sample
			int index = sampling.next();
			
			//Fetch sample
			Sample sample = dataset.getSample(index);
			Tensor input = sample.getInput();
			Tensor target = sample.getOutput();

			// Forward
			Tensor output = nn.forward(input, String.valueOf(index));
			
			// Error
			err += criterion.error(output, target).get(0);
			
			// Error gradient
			Tensor gradOut = criterion.grad(output, target);
			
			// Backward
			Tensor gradIn = nn.backward(gradOut, String.valueOf(index));
			
			// Accumulate gradient weights
			nn.getTrainables().values().stream().forEach(Trainable::accGradParameters);
		}
		
		if(batchAverage) {
			nn.getTrainables().values().stream().forEach(m -> {
				Tensor deltaParams = m.getDeltaParameters();
	
				factory.getTensorMath().div(deltaParams, deltaParams, batchSize);
						
				// Set DeltaParameters to be sure in case of remote module instance
				m.setDeltaParameters(deltaParams);
			});
			
			err /= batchSize;
		}
		
		// Run gradient processors
		gradientProcessor.calculateDelta(i);
		
		// Update parameters
		nn.getTrainables().values().stream().forEach(Trainable::updateParameters);
		
		return err;
	}
	
}
