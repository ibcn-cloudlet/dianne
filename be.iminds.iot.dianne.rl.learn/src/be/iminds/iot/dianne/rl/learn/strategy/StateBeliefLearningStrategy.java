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
import be.iminds.iot.dianne.api.rl.dataset.BatchedExperiencePoolSequence;
import be.iminds.iot.dianne.api.rl.dataset.ExperiencePool;
import be.iminds.iot.dianne.nn.learn.criterion.CriterionFactory;
import be.iminds.iot.dianne.nn.learn.criterion.CriterionFactory.CriterionConfig;
import be.iminds.iot.dianne.nn.learn.processors.ProcessorFactory;
import be.iminds.iot.dianne.nn.util.DianneConfigHandler;
import be.iminds.iot.dianne.rl.learn.strategy.config.StateBeliefConfig;
import be.iminds.iot.dianne.rnn.learn.sampling.SequenceSamplingFactory;
import be.iminds.iot.dianne.rnn.learn.sampling.SequenceSamplingStrategy;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;

/**
 * This strategy tries to train an unsupervized state representation from observations o_0 to o_T-1 by
 * decoding the current observation and predicting the next.
 * 
 * @author tverbele
 *
 */
public class StateBeliefLearningStrategy implements LearningStrategy {

	protected StateBeliefConfig config;
	
	protected ExperiencePool pool;
	protected SequenceSamplingStrategy sampling;
	protected int[] indices;
	
	protected BatchedExperiencePoolSequence sequence;
	
	protected NeuralNetwork prior;
	protected NeuralNetwork posterior;
	protected NeuralNetwork likelihood;
	
	protected UUID[] priorIn;
	protected UUID[] priorOut;

	protected UUID[] posteriorIn;
	protected UUID[] posteriorOut;
	
	protected Criterion reconCriterion;
	protected Criterion regulCriterion;
	
	protected GradientProcessor priorProcessor;
	protected GradientProcessor posteriorProcessor;
	protected GradientProcessor likelihoodProcessor;
	
	protected Tensor action;
	protected Tensor state;
	protected Tensor random;
	
	protected Tensor referencePriorParams;
	protected Tensor posteriorParamsGrad;
	
	protected List<Tensor> states = new ArrayList<>();
	protected List<Tensor> randoms = new ArrayList<>();
	
	@Override
	public void setup(Map<String, String> config, Dataset dataset, NeuralNetwork... nns) throws Exception {
		if(!(dataset instanceof ExperiencePool))
			throw new RuntimeException("Dataset is no experience pool");
		
		this.pool = (ExperiencePool) dataset;
		
		if(nns.length != 3)
			throw new RuntimeException("Invalid number of NN instances provided: "+nns.length+" (expected 3)");
			
		this.posterior = nns[0];
		this.likelihood = nns[1];
		this.prior = nns[2];
		
		this.posteriorIn = posterior.getModuleIds("State","Action","Observation");
		this.posteriorOut = new UUID[]{posterior.getOutput().getId()};
		
		this.priorIn = prior.getModuleIds("State","Action");
		this.priorOut = new UUID[]{prior.getOutput().getId()};
		
		this.config = DianneConfigHandler.getConfig(config, StateBeliefConfig.class);
		this.sampling = SequenceSamplingFactory.createSamplingStrategy(this.config.sampling, this.pool, config);
		// always sample from start index
		indices = new int[this.config.batchSize];
		for(int i=0;i<this.config.batchSize;i++){
			indices[i]=0;
		}
		
		this.reconCriterion = CriterionFactory.createCriterion(this.config.criterion, config);
		this.regulCriterion = CriterionFactory.createCriterion(CriterionConfig.GKL, config); 
		
		this.posteriorProcessor = ProcessorFactory.createGradientProcessor(this.config.method, posterior, config);
		this.likelihoodProcessor = ProcessorFactory.createGradientProcessor(this.config.method, likelihood, config);
		this.priorProcessor = ProcessorFactory.createGradientProcessor(this.config.method, prior, config);

		this.action = new Tensor(this.config.batchSize, this.pool.actionDims());
		this.state = new Tensor(this.config.batchSize, this.config.stateSize);
		this.random = new Tensor(this.config.batchSize, this.config.stateSize);

		this.posteriorParamsGrad = new Tensor(this.config.batchSize, 2*this.config.stateSize);
		this.referencePriorParams = new Tensor(this.config.batchSize, 2*this.config.stateSize);
		this.referencePriorParams.narrow(1, 0, this.config.stateSize).fill(0.0f);
		this.referencePriorParams.narrow(1, this.config.stateSize, this.config.stateSize).fill(1.0f);
		
		System.out.println("Start learning...");
	}

	@Override
	public LearnProgress processIteration(long i) throws Exception {
		// Reset deltas 
		prior.zeroDeltaParameters();
		posterior.zeroDeltaParameters();
		likelihood.zeroDeltaParameters();

		// Fetch sequence
		int[] seq = sampling.sequence(config.batchSize);
		sequence = pool.getBatchedSequence(sequence, seq, indices , config.sequenceLength);
		
		// Initial action/state
		// TODO: seeing as these are a valid state & action, this might not be desired?
		action.fill(0.0f);
		state.fill(0.0f);

		// Forward pass: generate sequence of state samples based on observations
		for(int t = 0; t < sequence.size(); t++){
			Tensor observation = sequence.getState(t);
			
			// Use approximate posterior to determine distribution parameters
			Tensor posteriorParams = posterior.forward(posteriorIn, posteriorOut, new Tensor[]{state, action, observation}).getValue().tensor;
			sampleState(state, posteriorParams, random);

			// Store for backward pass
			storeTensor(states, state, t);
			storeTensor(randoms, random, t);
			
			// Set next action (state already set)
			action = sequence.getAction(t);
		}
		
		// Keep separate reconstruction and regularization loss
		float reconLoss = 0;
		float regulLoss = 0;
		
		// Additional for final timestep: reconstruction loss on o_T-1
		Tensor reconParams = likelihood.forward(states.get(sequence.size()-1));
		Tensor observation = sequence.getState(sequence.size()-1);
		
		reconLoss += TensorOps.mean(reconCriterion.loss(reconParams, observation));
		Tensor stateGrad = likelihood.backward(reconCriterion.grad(reconParams, observation));
		likelihood.accGradParameters();
		
		// Other timesteps:
		// - Convert gradient of next state sample to posterior parameters
		// - Add regularization loss based on prior
		// - Calculate gradient to current state sample
		// - Add gradient from prior
		// - Add gradient from likelihood
		for(int t = sequence.size()-2; t >= 0; t--){
			// Convert gradient of s_t+1 to gradient on its posterior parameters
			stateDistributionGrad(posteriorParamsGrad, stateGrad, randoms.get(t+1));
			
			// Get prior and posterior parameters of s_t+1
			Tensor priorParams = prior.forward(priorIn, priorOut, new Tensor[]{states.get(t), sequence.getAction(t)}).getValue().tensor;
			Tensor posteriorParams = posterior.forward(posteriorIn, posteriorOut, new Tensor[]{states.get(t), sequence.getAction(t), sequence.getState(t+1)}).getValue().tensor;
			
			// Add regularization loss based on prior on s_t+1
			regulLoss += TensorOps.mean(regulCriterion.loss(posteriorParams, priorParams));
			TensorOps.add(posteriorParamsGrad, posteriorParamsGrad, regulCriterion.grad(posteriorParams, priorParams));
			
			// Calculate gradient to s_t from posterior of s_t+1
			stateGrad = posterior.backward(posteriorOut, posteriorIn, new Tensor[]{posteriorParamsGrad}).getValue().tensors.get(posteriorIn[0]);
			posterior.accGradParameters();
			
			// Calculate gradient to s_t from prior of s_t+1
			Tensor priorParamsGrad = regulCriterion.gradTarget(posteriorParams, priorParams);
			// Optional prior regularization
			if(config.priorRegularization > 0) {
				regulLoss += config.priorRegularization*TensorOps.mean(regulCriterion.loss(priorParams, referencePriorParams));
				TensorOps.add(priorParamsGrad, priorParamsGrad, config.priorRegularization, regulCriterion.grad(priorParams, referencePriorParams));
			}
			TensorOps.add(stateGrad, stateGrad, prior.backward(priorOut, priorIn, new Tensor[]{priorParamsGrad}).getValue().tensors.get(priorIn[0]));
			prior.accGradParameters();
			
			// Add gradient from likelihood of o_t
			reconParams = likelihood.forward(states.get(t));
			observation = sequence.getState(t);
			
			reconLoss += TensorOps.mean(reconCriterion.loss(reconParams, observation));
			
			Tensor reconParamsGrad = reconCriterion.grad(reconParams, observation);
			TensorOps.add(stateGrad, stateGrad, likelihood.backward(reconParamsGrad));
			likelihood.accGradParameters();
		}
		
		// Additional for first timestep: loss on posterior and prior of s_0
		action.fill(0.0f);
		state.fill(0.0f);
		
		// Convert gradient of s_0 to gradient on its posterior parameters
		stateDistributionGrad(posteriorParamsGrad, stateGrad, randoms.get(0));
		
		// Get prior and posterior parameters of s_0
		Tensor priorParams = prior.forward(priorIn, priorOut, new Tensor[]{state, action}).getValue().tensor;
		Tensor posteriorParams = posterior.forward(posteriorIn, posteriorOut, new Tensor[]{state, action, sequence.getState(0)}).getValue().tensor;
		
		// Add regularization loss based on prior on s_0
		regulLoss += TensorOps.mean(regulCriterion.loss(posteriorParams, priorParams));
		TensorOps.add(posteriorParamsGrad, posteriorParamsGrad, regulCriterion.grad(posteriorParams, priorParams));
		
		// Calculate gradient of posterior of s_0
		posterior.backward(posteriorOut, posteriorIn, new Tensor[]{posteriorParamsGrad}).getValue();
		posterior.accGradParameters();
		
		// Calculate gradient of prior of s_0
		Tensor priorParamsGrad = regulCriterion.gradTarget(posteriorParams, priorParams);
		// Optional prior regularization
		if(config.priorRegularization > 0) {
			regulLoss += config.priorRegularization*TensorOps.mean(regulCriterion.loss(priorParams, referencePriorParams));
			TensorOps.add(priorParamsGrad, priorParamsGrad, config.priorRegularization, regulCriterion.grad(priorParams, referencePriorParams));
		}
		prior.backward(priorOut, priorIn, new Tensor[]{priorParamsGrad}).getValue();
		prior.accGradParameters();

		// Calculate the deltas
		priorProcessor.calculateDelta(i);
		posteriorProcessor.calculateDelta(i);
		likelihoodProcessor.calculateDelta(i);
		
		// Update the parameters
		prior.updateParameters();
		posterior.updateParameters();
		likelihood.updateParameters();
		
		return new LearnProgress(i, (reconLoss+regulLoss)/sequence.size);
	}
	
	private static void storeTensor(List<Tensor> list, Tensor tensor, int index){
		while(index>=list.size()){
			list.add(new Tensor());
		}
		tensor.copyInto(list.get(index));
	}
	
	private void sampleState(Tensor state, Tensor stateDistribution, Tensor random) {
		// latentParam => latent
		Tensor means = stateDistribution.narrow(1, 0, config.stateSize);
		Tensor stdevs = stateDistribution.narrow(1, config.stateSize, config.stateSize);
		
		random.randn();
		
		TensorOps.cmul(state, random, stdevs);
		TensorOps.add(state, state, means);
	}
	
	private void stateDistributionGrad(Tensor stateDistributionGrad, Tensor stateGrad, Tensor random) {
		// latentGrad => latentParamsGrad
		Tensor gradMeans = stateDistributionGrad.narrow(1, 0, config.stateSize);
		Tensor gradStdevs = stateDistributionGrad.narrow(1, config.stateSize, config.stateSize);
		
		stateGrad.copyInto(gradMeans);
		TensorOps.cmul(gradStdevs, stateGrad, random);
	}
}
