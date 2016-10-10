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
package be.iminds.iot.dianne.rl.learn.strategy;

import java.util.Map;

import be.iminds.iot.dianne.api.dataset.Batch;
import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.nn.NeuralNetwork;
import be.iminds.iot.dianne.api.nn.learn.Criterion;
import be.iminds.iot.dianne.api.nn.learn.GradientProcessor;
import be.iminds.iot.dianne.api.nn.learn.LearnProgress;
import be.iminds.iot.dianne.api.nn.learn.LearningStrategy;
import be.iminds.iot.dianne.api.nn.learn.SamplingStrategy;
import be.iminds.iot.dianne.api.rl.dataset.ExperiencePool;
import be.iminds.iot.dianne.api.rl.dataset.ExperiencePoolSample;
import be.iminds.iot.dianne.api.rl.learn.QLearnProgress;
import be.iminds.iot.dianne.nn.learn.criterion.CriterionFactory;
import be.iminds.iot.dianne.nn.learn.processors.ProcessorFactory;
import be.iminds.iot.dianne.nn.learn.sampling.SamplingFactory;
import be.iminds.iot.dianne.nn.util.DianneConfigHandler;
import be.iminds.iot.dianne.rl.learn.strategy.config.DeepQConfig;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;

/**
 * This learning strategy implements DQN
 * 
 * The strategy requires 2 NN instances of the same NN: one acting as a target for the other
 * 
 * In order to make this work, make sure to set the syncInterval of the target to make sure it 
 * updates from time to time to the weights of the trained NN.
 * 
 * @author tverbele
 *
 */
public class DeepQLearningStrategy implements LearningStrategy {

	protected ExperiencePool pool; 
	protected NeuralNetwork nn;
	protected NeuralNetwork target;
	
	protected DeepQConfig config;
	protected GradientProcessor gradientProcessor;
	protected Criterion criterion;
	protected SamplingStrategy sampling;
	
	protected Batch batch;
	
	@Override
	public void setup(Map<String, String> config, Dataset dataset, NeuralNetwork... nns) throws Exception {
		if(!(dataset instanceof ExperiencePool))
			throw new RuntimeException("Dataset is no experience pool");
		
		this.pool = (ExperiencePool)dataset;
		
		if(nns.length != 2)
			throw new RuntimeException("Invalid number of NN instances provided: "+nns.length+" ( expected 2 )");
			
		this.nn = nns[0];
		this.target = nns[1];
		
		this.config = DianneConfigHandler.getConfig(config, DeepQConfig.class);
		sampling = SamplingFactory.createSamplingStrategy(this.config.sampling, dataset, config);
		criterion = CriterionFactory.createCriterion(this.config.criterion, config);
		gradientProcessor = ProcessorFactory.createGradientProcessor(this.config.method, nn, config);
		
		// wait until experience pool has a sufficient amount of samples
		if(pool.size() < this.config.minSamples){
			System.out.println("Experience pool has too few samples, waiting a bit to start learning...");
			while(pool.size() < this.config.minSamples){
				try {
					Thread.sleep(5000);
				} catch (InterruptedException e) {
					return;
				}
			}
		}
		
		System.out.println("Start learning...");
	}

	@Override
	public LearnProgress processIteration(long i) throws Exception {
		nn.zeroDeltaParameters();

		float error = 0;
		float q = 0;
			
		// TODO actually process in batch
		for(int k=0;k<config.batchSize;k++){
			// new sample
			int index = sampling.next();
			
			ExperiencePoolSample sample = pool.getSample(index);
			
			Tensor in = sample.input;

			// forward
			Tensor out = nn.forward(in, ""+index);
			
			// evaluate criterion
			Tensor action = sample.target;
			float reward = sample.reward;
			Tensor nextState = sample.nextState;
			
			float targetQ = 0;
			
			if(sample.isTerminal){
				// terminal state
				targetQ = reward;
			} else {
				Tensor nextQ = target.forward(nextState, ""+index);
				targetQ = reward + config.discount * TensorOps.max(nextQ);
			}
			
			Tensor targetOut = out.copyInto(null);
			targetOut.set(targetQ, TensorOps.argmax(action));
			
			q += out.get(TensorOps.argmax(action));
		
			Tensor e = criterion.loss(out, targetOut);
			error += e.get(0);
			
			Tensor gradOut = criterion.grad(out, targetOut);
			
			// backward
			Tensor gradIn = nn.backward(gradOut, ""+index);
			
			// acc gradParameters
			nn.accGradParameters();
		}
		
		gradientProcessor.calculateDelta(i);

		nn.updateParameters();
		
		return new QLearnProgress(i, error/config.batchSize, q/config.batchSize);
	}

}
