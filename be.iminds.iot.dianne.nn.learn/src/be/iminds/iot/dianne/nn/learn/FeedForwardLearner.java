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

import be.iminds.iot.dianne.api.nn.learn.Learner;
import be.iminds.iot.dianne.tensor.Tensor;

@Component(service=Learner.class, property={"aiolos.unique=true"})
public class FeedForwardLearner extends AbstractLearner {
	
	protected int batchSize = 10;

	protected void loadConfig(Map<String, String> config){
		super.loadConfig(config);
		
		if(config.get("batchSize")!=null){
			batchSize = Integer.parseInt(config.get("batchSize"));
		}
		System.out.println("* batchSize = " +batchSize);

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
			Tensor in = dataset.getInputSample(index);

			// forward
			Tensor out = nn.forward(in, ""+index);
			
			// evaluate criterion
			Tensor e = criterion.error(out, dataset.getOutputSample(index));
			err += e.get(0);
			Tensor gradOut = criterion.grad(out, dataset.getOutputSample(index));
			
			// backward
			Tensor gradIn = nn.backward(gradOut, ""+index);
			
			// acc gradParameters
			nn.getTrainables().values().stream().forEach(m -> m.accGradParameters());
		}
		
		// run gradient processors
		gradientProcessor.calculateDelta(i);
		
		// update parameters
		nn.getTrainables().entrySet().stream().forEach(e -> {
			e.getValue().updateParameters();
		});
		
		return err/batchSize;
	}
	
}
