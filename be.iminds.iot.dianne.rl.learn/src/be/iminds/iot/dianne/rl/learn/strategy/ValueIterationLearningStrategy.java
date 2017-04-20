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
import java.util.UUID;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.nn.NeuralNetwork;
import be.iminds.iot.dianne.api.nn.learn.Criterion;
import be.iminds.iot.dianne.api.nn.learn.GradientProcessor;
import be.iminds.iot.dianne.api.nn.learn.LearnProgress;
import be.iminds.iot.dianne.api.nn.learn.LearningStrategy;
import be.iminds.iot.dianne.nn.learn.criterion.CriterionFactory;
import be.iminds.iot.dianne.nn.learn.processors.ProcessorFactory;
import be.iminds.iot.dianne.nn.util.DianneConfigHandler;
import be.iminds.iot.dianne.rl.learn.strategy.config.ValueIterationConfig;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;

/**
 * This learning strategy implements a form of value iterations
 * 
 * The strategy requires 4 NN instances: a model of the transition distribution, reward likelihood
 * and 2 NN instances of the same Q value NN: one acting as a target for the other
 * 
 * In order to make this work, make sure to set the syncInterval of the target to make sure it 
 * updates from time to time to the weights of the trained NN.
 * 
 * @author sbohez
 *
 */
public class ValueIterationLearningStrategy implements LearningStrategy {

	protected ValueIterationConfig config;
	
	protected NeuralNetwork valueNetwork;
	protected NeuralNetwork targetNetwork;
	protected NeuralNetwork stateModel;
	protected NeuralNetwork rewardModel;
	
	protected UUID[] stateModelIn;
	protected UUID[] stateModelOut;
	
	protected UUID[] rewardModelIn;
	protected UUID[] rewardModelOut;
	
	protected Criterion criterion;
	protected GradientProcessor gradientProcessor;
	
	protected Tensor action;
	protected Tensor stateSample;
	protected Tensor random4state;
	protected Tensor rewardSample;
	protected Tensor random4reward;
	protected Tensor targetValue;
	
	protected List<Tensor> states = new ArrayList<>();
	protected List<Tensor> actions = new ArrayList<>();
	protected List<Tensor> rewards = new ArrayList<>();
	
	@Override
	public void setup(Map<String, String> config, Dataset dataset, NeuralNetwork... nns) throws Exception {		
		if(nns.length != 4)
			throw new RuntimeException("Invalid number of NN instances provided: "+nns.length+" (expected 4)");
			
		this.valueNetwork = nns[0];
		this.targetNetwork = nns[1];
		this.stateModel = nns[2];
		this.rewardModel = nns[3];
		
		this.stateModelIn = stateModel.getModuleIds("State","Action");
		this.stateModelOut = new UUID[]{stateModel.getOutput().getId()};
		
		this.rewardModelIn = rewardModel.getModuleIds("State","Action");
		this.rewardModelOut = new UUID[]{rewardModel.getOutput().getId()};
		
		this.config = DianneConfigHandler.getConfig(config, ValueIterationConfig.class);
		this.criterion = CriterionFactory.createCriterion(this.config.criterion, config);
		this.gradientProcessor = ProcessorFactory.createGradientProcessor(this.config.method, valueNetwork, config);
		
		// Pre-allocate tensors for batch operations
		this.action = new Tensor(this.config.batchSize, this.config.actionDims);
		this.stateSample = new Tensor(this.config.batchSize, this.config.stateDims);
		this.random4state = new Tensor(this.config.batchSize, this.config.stateDims);
		this.rewardSample = new Tensor(this.config.batchSize);
		this.random4reward = new Tensor(this.config.batchSize);
		this.targetValue = new Tensor(this.config.batchSize, this.config.actionDims);
		
		System.out.println("Start learning...");
	}

	@Override
	public LearnProgress processIteration(long i) throws Exception {
		valueNetwork.zeroDeltaParameters();
		
		action.fill(0);
		stateSample.fill(0);
		
		for(int t = 0; t < (config.episodeTerminal ? config.episodeLength : config.episodeLength+config.Qsteps); t++) {
			Tensor stateDistribution = stateModel.forward(stateModelIn, stateModelOut, new Tensor[]{stateSample, action}).getValue().tensor;
			sample(stateSample, stateDistribution, random4state);
			storeTensor(states, stateSample, t);
			
			Tensor value = config.doubleQ ? valueNetwork.forward(stateSample) : targetNetwork.forward(stateSample);
			
			action.fill(0);
			for(int b = 0; b < config.batchSize; b++) {
				action.set(1, b, TensorOps.argmax(value.select(0, b)));
			}
			storeTensor(actions, action, t);
			
			Tensor rewardDistribution = rewardModel.forward(rewardModelIn, rewardModelOut, new Tensor[]{stateSample, action}).getValue().tensor;
			sample(rewardSample, rewardDistribution, random4reward);
			storeTensor(rewards, rewardSample, t);
		}
		
		float loss = 0, value = 0;
		
		for(int t = 0; t < config.episodeLength; t++) {
			Tensor currentState = states.get(t);
			Tensor currentAction = actions.get(t);
			Tensor currentValue = valueNetwork.forward(currentState);
			TensorOps.cmul(currentValue, currentValue, currentAction);
			value += TensorOps.sum(currentValue)/config.batchSize;
			
			// Reusing reward sample to calculate target value
			if(config.Qsteps > 0 && (config.episodeTerminal ? t < config.episodeLength-config.Qsteps : true)) {
				Tensor futureState = states.get(t+config.Qsteps);
				Tensor futureAction = actions.get(t+config.Qsteps); //doubleQ already included
				Tensor futureValue = targetNetwork.forward(futureState);
				TensorOps.cmul(futureValue, futureValue, futureAction);
				
				for(int b = 0; b < config.batchSize; b++) {
					rewardSample.set(TensorOps.sum(futureValue.select(0, b)), b);
				}
			} else {
				rewardSample.fill(0);
			}
			
			for(int n = config.episodeTerminal && t >= config.episodeLength-config.Qsteps ? config.episodeLength-1 : t+config.Qsteps-1; n >= t; n--) {
				TensorOps.mul(rewardSample, rewardSample, config.discount);
				TensorOps.add(rewardSample, rewardSample, rewards.get(n));
			}
			
			targetValue.fill(0);
			for(int b = 0; b < config.batchSize; b++) {
				Tensor v = targetValue.select(0, b);
				Tensor a = currentAction.select(0,b);
				TensorOps.mul(v, a, rewardSample.get(b));
			}
			
			loss += TensorOps.mean(criterion.loss(currentValue, targetValue));
			valueNetwork.backward(criterion.grad(currentValue, targetValue), true);
		}
		
		gradientProcessor.calculateDelta(i);
		valueNetwork.updateParameters();

		return new LearnProgress(i, loss/config.episodeLength, new String[]{"q"}, new float[]{value/config.episodeLength});
	}
	
	private static void storeTensor(List<Tensor> list, Tensor tensor, int index){
		while(index>=list.size()){
			list.add(new Tensor());
		}
		tensor.copyInto(list.get(index));
	}

	private static void sample(Tensor sample, Tensor distribution, Tensor random) {
		int size = distribution.size(1)/2;
		Tensor means = distribution.narrow(1, 0, size);
		Tensor stdevs = distribution.narrow(1, size, size);
		
		random.randn();
		
		TensorOps.cmul(sample, random, stdevs);
		TensorOps.add(sample, sample, means);
	}
}
