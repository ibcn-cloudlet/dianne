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

import java.lang.reflect.InvocationTargetException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.osgi.service.component.annotations.Component;
import org.osgi.util.promise.Promise;

import be.iminds.iot.dianne.api.dataset.Batch;
import be.iminds.iot.dianne.api.nn.learn.Learner;
import be.iminds.iot.dianne.api.nn.module.Trainable;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;

@Component(service=Learner.class, 
	property={"aiolos.unique=true",
			  "dianne.learner.category=FF"})
public class FeedForwardLearner extends AbstractLearner {
	
	protected int batchSize = 10;
	protected boolean batchAverage = true;
	
	// For loading next batch while processing current batch
	private Batch batch = null;
	private Batch nextBatch = null;
	
	protected void loadConfig(Map<String, String> config){
		super.loadConfig(config);
		
		if(config.containsKey("batchSize"))
			batchSize = Integer.parseInt(config.get("batchSize"));
		
		if(config.containsKey("batchAverage"))
			batchAverage = Boolean.parseBoolean(config.get("batchAverage"));
		
		System.out.println("* batchSize = " +batchSize);
		System.out.println("* batchAverage = " + batchAverage);
		
		// Reset next batch
		batch = null;
		nextBatch = null;
	}
	
	private void loadBatch(){
		int[] indices = new int[batchSize];
		
		// Select new samples
		for(int k=0;k<batchSize;k++)
			indices[k] = sampling.next();

		nextBatch = dataset.getBatch(nextBatch, indices);
	}
	
	protected float process(long i){
		// Clear delta params
		nn.getTrainables().values().stream().forEach(Trainable::zeroDeltaParameters);
		
		// Use a placeholder for error
		final float[] error = new float[1];
		
		// Load batch if necessary
		if(nextBatch==null)
			loadBatch();
		
		// Flip current/next
		Batch temp = batch;
		batch = nextBatch;
		nextBatch = temp;
		
		// Forward/backward pass - executed asynchronously
		if(batch.input != null) {
			// Execute in batch
			Promise result = nn.forward(null, null, batch.input).then(
					p -> {
						// Forward
						Tensor output = p.getValue().tensor;
						
						// Error
						error[0] += criterion.error(output, batch.output).get(0);
	
						// Error gradient
						Tensor gradOut = criterion.grad(output, batch.output);
						
						// Backward
						return nn.backward(null, null, gradOut);
					}).then(		
					p -> {	
						// Accumulate gradient weights
						nn.getTrainables().values().stream().forEach(Trainable::accGradParameters);
						
						return p;
					});
			
			// Load next batch while processing previous one
			loadBatch();
			
			try {
				result.getValue();
			} catch (InvocationTargetException | InterruptedException e) {
				e.printStackTrace();
			}
		} else {
			// Cannot load a batch for this dataset, still process one by one
			for(int k=0;k<batchSize;k++){
				final int b = k;
				Promise result = nn.forward(null, null, batch.inputSamples[b]).then(
						p -> {
							// Forward
							Tensor output = p.getValue().tensor;
							
							// Error
							error[0] += criterion.error(output, batch.outputSamples[b]).get(0);
		
							// Error gradient
							Tensor gradOut = criterion.grad(output, batch.outputSamples[b]);
							
							// Backward
							return nn.backward(null, null, gradOut);
						}).then(
						p -> {	
							// Accumulate gradient weights
							nn.getTrainables().values().stream().forEach(Trainable::accGradParameters);
							
							return p;
						});
				
				try {
					result.getValue();
				} catch (InvocationTargetException | InterruptedException e) {
					e.printStackTrace();
				}
			}
			
			// Load next batch
			loadBatch();
		}

		// Batch done, calculate deltas
		if(batchAverage) {
			nn.getTrainables().values().stream().forEach(m -> {
				Tensor deltaParams = m.getDeltaParameters();
	
				TensorOps.div(deltaParams, deltaParams, batchSize);
		
				// Set DeltaParameters to be sure in case of remote module instance
				m.setDeltaParameters(deltaParams);
			});
			
			error[0] /= batchSize;
		}
		
		// Run gradient processors
		gradientProcessor.calculateDelta(i);
		
		// Update parameters
		nn.getTrainables().values().stream().forEach(Trainable::updateParameters);

		return error[0];
	}
	
}
