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

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.nn.NeuralNetwork;
import be.iminds.iot.dianne.api.nn.learn.GradientProcessor;
import be.iminds.iot.dianne.api.nn.learn.LearnProgress;
import be.iminds.iot.dianne.api.nn.learn.LearningStrategy;
import be.iminds.iot.dianne.api.rl.dataset.BatchedExperiencePoolSequence;
import be.iminds.iot.dianne.api.rl.dataset.ExperiencePool;
import be.iminds.iot.dianne.api.rl.dataset.ExperiencePoolBatch;
import be.iminds.iot.dianne.api.rl.learn.QLearnProgress;
import be.iminds.iot.dianne.nn.learn.processors.ProcessorFactory;
import be.iminds.iot.dianne.nn.util.DianneConfigHandler;
import be.iminds.iot.dianne.rl.learn.strategy.config.RecurrentDeepQConfig;
import be.iminds.iot.dianne.rnn.learn.criterion.SequenceCriterion;
import be.iminds.iot.dianne.rnn.learn.criterion.SequenceCriterionFactory;
import be.iminds.iot.dianne.rnn.learn.sampling.SequenceSamplingFactory;
import be.iminds.iot.dianne.rnn.learn.sampling.SequenceSamplingStrategy;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;

/**
 * This learning strategy implements DQN with recurrent layer
 * 
 * The strategy requires 2 NN instances of the same NN: one acting as a target for the other
 * 
 * @author tverbele
 *
 */
public class RecurrentDeepQLearningStrategy implements LearningStrategy {

	protected RecurrentDeepQConfig config;
	
	protected ExperiencePool pool;
	protected SequenceSamplingStrategy sampling;
	
	protected ExperiencePoolBatch batch;
	
	protected NeuralNetwork valueNetwork;
	protected NeuralNetwork targetNetwork;
	
	protected SequenceCriterion criterion;
	protected GradientProcessor gradientProcessor;
	
	protected Tensor actionBatch;
	protected List<Tensor> targets;
	
	protected BatchedExperiencePoolSequence sequence;
	
	@Override
	public void setup(Map<String, String> config, Dataset dataset, NeuralNetwork... nns) throws Exception {
		if(!(dataset instanceof ExperiencePool))
			throw new RuntimeException("Dataset is no experience pool");
		
		this.pool = (ExperiencePool) dataset;
		
		if(nns.length != 2)
			throw new RuntimeException("Invalid number of NN instances provided: "+nns.length+" (expected 2)");
			
		this.valueNetwork = nns[0];
		this.targetNetwork = nns[1];
		
		this.config = DianneConfigHandler.getConfig(config, RecurrentDeepQConfig.class);
		this.sampling = SequenceSamplingFactory.createSamplingStrategy(this.config.sampling, this.pool, config);
		this.criterion = SequenceCriterionFactory.createCriterion(this.config.criterion, config);
		this.gradientProcessor = ProcessorFactory.createGradientProcessor(this.config.method, valueNetwork, config);
		
		this.actionBatch = new Tensor(this.config.batchSize, this.pool.actionDims()[0]);
		this.targets = new ArrayList<Tensor>();
		for(int i=0;i<this.config.sequenceLength;i++){
			targets.add(new Tensor(this.config.batchSize, this.pool.actionDims()[0]));
		}

		
		// Wait for the pool to contain enough samples
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
		// Reset the deltas
		valueNetwork.zeroDeltaParameters();
		
		// Reset hidden states
		this.valueNetwork.resetMemory(this.config.batchSize);
		this.targetNetwork.resetMemory(this.config.batchSize);
		
		// sample sequence
		int[] seq = sampling.sequence(config.batchSize);
		int[] index = sampling.next(seq, config.sequenceLength);
		sequence = pool.getBatchedSequence(sequence, seq, index, config.sequenceLength);
		
		List<Tensor> nextValues = targetNetwork.forward(sequence.getNextStates());
		
		for(int s=0; s<config.sequenceLength; s++){
			Tensor target = targets.get(s);
			target.fill(0.0f);
			for(int b=0; b<config.batchSize; b++){
				int action = TensorOps.argmax(sequence.get(s).getAction(b));
				float reward = sequence.get(s).getScalarReward(b);
				
				// Calculate the target value
				if(!sequence.get(s).isTerminal(b)) {
					// TODO add double q learning?
					
					// Set the target value using the Bellman equation
					target.set(reward + config.discount*TensorOps.max(nextValues.get(s).select(0, b)), b, action);
				} else {
					// If the next state is terminal, the target value is equal to the reward
					target.set(reward, b, action);
				}
			}
		}
		
		List<Tensor> values = valueNetwork.forward(sequence.getStates());
		
		float value = 0;
		for(int s=0; s<config.sequenceLength; s++){
			for(int b = 0; b < config.batchSize; b++) {
				value += TensorOps.max(values.get(s).select(0, b));
			}
		}
		value /= config.batchSize;
		value /= config.sequenceLength;
		
		for(int s=0; s<config.sequenceLength; s++){
			TensorOps.cmul(values.get(s), values.get(s), sequence.getAction(s));
		}
		
		float loss =  TensorOps.mean(criterion.loss(values, targets).stream().reduce((t1,t2) -> TensorOps.add(t1, t1, t2)).get())/sequence.size;
		List<Tensor> grad = criterion.grad(values, targets);
		
		// Backward pass of the critic
		valueNetwork.backward(grad, true);
		
		// Call the processors to set the updates
		gradientProcessor.calculateDelta(i);
		
		// Apply the updates
		// Note: target network gets updated automatically by setting the syncInterval option
		valueNetwork.updateParameters();
		
		
		return new QLearnProgress(i, loss, value);
	}

}
