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
		// clear delta params
		nn.getTrainables().entrySet().stream().forEach(e -> {
			e.getValue().zeroDeltaParameters();
		});
		
		// calculate grad
		float err = 0;
		for(int k=0;k<batchSize;k++){
			// new sample
			int index = sampling.next();
			Sample sample = dataset.getSample(index);
			Tensor in = sample.input;

			// forward
			Tensor out = nn.forward(in, ""+index);
			
			// evaluate criterion
			Tensor e = criterion.error(out, sample.output);
			err += e.get(0);
			Tensor gradOut = criterion.grad(out, sample.output);
			
			// backward
			Tensor gradIn = nn.backward(gradOut, ""+index);
			
			// acc gradParameters
			nn.getTrainables().values().stream().forEach(m -> m.accGradParameters());
		}
		
		if(batchAverage) {
			nn.getTrainables().values().stream().forEach(m -> {
				Tensor deltaParams = m.getDeltaParameters();
	
				factory.getTensorMath().div(deltaParams, deltaParams, batchSize);
						
				// set DeltaParameters to be sure in case of remote module instance
				m.setDeltaParameters(deltaParams);
			});
			
			err /= batchSize;
		}
		
		// run gradient processors
		gradientProcessor.calculateDelta(i);
		
		// update parameters
		nn.getTrainables().values().stream().forEach(Trainable::updateParameters);
		
		return err;
	}
	
}
