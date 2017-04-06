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
import java.util.Arrays;
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
	protected NeuralNetwork observationLikelihood;
	protected NeuralNetwork rewardLikelihood;
	
	protected UUID[] priorIn;
	protected UUID[] priorOut;

	protected UUID[] posteriorIn;
	protected UUID[] posteriorOut;
	
	protected UUID[] rewardLikelihoodIn;
	protected UUID[] rewardLikelihoodOut;
	
	protected Criterion priorRegulCriterion;
	protected Criterion posteriorRegulCriterion;
	protected Criterion observationReconCriterion;
	protected Criterion rewardReconCriterion;
	
	protected GradientProcessor priorProcessor;
	protected GradientProcessor posteriorProcessor;
	protected GradientProcessor observationLikelihoodProcessor;
	protected GradientProcessor rewardLikelihoodProcessor;
	
	protected Tensor action;
	protected Tensor state;
	protected Tensor random;
	
	protected Tensor sampleParamsGrad;
	protected Tensor referencePriorParams;
	
	protected boolean[] dropped;
	
	protected List<Tensor> states = new ArrayList<>();
	protected List<Tensor> randoms = new ArrayList<>();
	
	@Override
	public void setup(Map<String, String> config, Dataset dataset, NeuralNetwork... nns) throws Exception {
		if(!(dataset instanceof ExperiencePool))
			throw new RuntimeException("Dataset is no experience pool");
		
		this.pool = (ExperiencePool) dataset;
		
		if(nns.length != 3 && nns.length != 4)
			throw new RuntimeException("Invalid number of NN instances provided: "+nns.length+" (expected 3 or 4)");
			
		this.prior = nns[0];
		this.posterior = nns[1];
		this.observationLikelihood = nns[2];
		this.rewardLikelihood = nns.length == 4 ? nns[3] : null;
		
		this.posteriorIn = posterior.getModuleIds("State","Action","Observation");
		this.posteriorOut = new UUID[]{posterior.getOutput().getId()};
		
		this.priorIn = prior.getModuleIds("State","Action");
		this.priorOut = new UUID[]{prior.getOutput().getId()};
		
		this.rewardLikelihoodIn = rewardLikelihood != null ? rewardLikelihood.getModuleIds("State","Action") : null;
		this.rewardLikelihoodOut = rewardLikelihood != null ? new UUID[]{rewardLikelihood.getOutput().getId()} : null;
		
		this.config = DianneConfigHandler.getConfig(config, StateBeliefConfig.class);
		this.sampling = SequenceSamplingFactory.createSamplingStrategy(this.config.sampling, this.pool, config);
		
		// Always sample from start index as we want to learn prior over initial states, not all states
		indices = new int[this.config.batchSize];
		Arrays.fill(indices, 0);
		
		this.priorRegulCriterion = CriterionFactory.createCriterion(CriterionConfig.GKL, config);
		this.posteriorRegulCriterion = CriterionFactory.createCriterion(CriterionConfig.GKL, config);
		this.observationReconCriterion = CriterionFactory.createCriterion(this.config.criterion, config);
		this.rewardReconCriterion = rewardLikelihood != null ? CriterionFactory.createCriterion(CriterionConfig.GAU, config) : null;
		
		this.posteriorProcessor = ProcessorFactory.createGradientProcessor(this.config.method, posterior, config);
		this.priorProcessor = ProcessorFactory.createGradientProcessor(this.config.method, prior, config);
		this.observationLikelihoodProcessor = ProcessorFactory.createGradientProcessor(this.config.method, observationLikelihood, config);
		this.rewardLikelihoodProcessor = rewardLikelihood != null ? ProcessorFactory.createGradientProcessor(this.config.method, rewardLikelihood, config) : null;

		this.action = new Tensor(this.config.batchSize, this.pool.actionDims());
		this.state = new Tensor(this.config.batchSize, this.config.stateSize);
		this.random = new Tensor(this.config.batchSize, this.config.stateSize);

		this.sampleParamsGrad = new Tensor(this.config.batchSize, 2*this.config.stateSize);
		this.referencePriorParams = new Tensor(this.config.batchSize, 2*this.config.stateSize);
		this.referencePriorParams.narrow(1, 0, this.config.stateSize).fill(0.0f);
		this.referencePriorParams.narrow(1, this.config.stateSize, this.config.stateSize).fill(1.0f);
		
		this.dropped = new boolean[this.config.sequenceLength];
		
		System.out.println("Start learning...");
	}

	@Override
	public LearnProgress processIteration(long i) throws Exception {
		// Reset deltas 
		prior.zeroDeltaParameters();
		posterior.zeroDeltaParameters();
		observationLikelihood.zeroDeltaParameters();
		if(rewardLikelihood != null)
			rewardLikelihood.zeroDeltaParameters();
		
		// Reset dropped
		Arrays.fill(dropped, false);

		// Fetch sequence
		int[] seq = sampling.sequence(config.batchSize);
		sequence = pool.getBatchedSequence(sequence, seq, indices , config.sequenceLength);
		
		// Initial action/state
		// TODO: seeing as these are in principle a valid state & action, will this work?
		action.fill(0.0f);
		state.fill(0.0f);

		// Forward pass: generate sequence of state samples based on observations
		for(int t = 0; t < sequence.size(); t++){
			Tensor observation = sequence.getState(t);
			
			// Randomly drop observation
			if(dropped[t] = Math.random() < config.dropRate) {
				// Use prior to determine distribution parameters
				Tensor priorParams = prior.forward(priorIn, priorOut, new Tensor[]{state, action}).getValue().tensor;
				sampleState(state, priorParams, random);
			} else {
				// Use approximate posterior to determine distribution parameters
				Tensor posteriorParams = posterior.forward(posteriorIn, posteriorOut, new Tensor[]{state, action, observation}).getValue().tensor;
				sampleState(state, posteriorParams, random);
			}

			// Store for backward pass
			storeTensor(states, state, t);
			storeTensor(randoms, random, t);
			
			// Set next action (state already set)
			action = sequence.getAction(t);
		}
		
		// Keep separate regularization and reconstruction loss
		float priorRegulLoss = 0, posteriorRegulLoss = 0;
		float observationReconLoss = 0, rewardReconLoss = 0;
		
		Tensor stateGrad;
		if(!dropped[sequence.size()-1]) {
			// Additional for final timestep: reconstruction loss on o_T-1
			Tensor observationReconParams = observationLikelihood.forward(states.get(sequence.size()-1));
			Tensor observation = sequence.getState(sequence.size()-1);
			
			observationReconLoss += TensorOps.mean(observationReconCriterion.loss(observationReconParams, observation));
			stateGrad = observationLikelihood.backward(observationReconCriterion.grad(observationReconParams, observation),true);
		} else {
			stateGrad = new Tensor(this.config.batchSize, this.config.stateSize);
			stateGrad.fill(0.0f);
		}
		
		if(rewardLikelihood != null) {
			// Optional: reconstruction loss on r_T-1
			Tensor rewardReconParams = rewardLikelihood.forward(rewardLikelihoodIn, rewardLikelihoodOut, new Tensor[]{states.get(sequence.size()-1), sequence.getAction(sequence.size()-1)}).getValue().tensor;
			Tensor reward = sequence.getReward(sequence.size()-1);
			Tensor terminal = sequence.getTerminal(sequence.size()-1);
			terminal.reshape(config.batchSize);
			
			Tensor loss = rewardReconCriterion.loss(rewardReconParams, reward);
			TensorOps.cmul(loss, loss, terminal);
			rewardReconLoss += TensorOps.mean(loss);
			
			Tensor grad = rewardReconCriterion.grad(rewardReconParams, reward);
			rmul(grad, grad, terminal);
			TensorOps.add(stateGrad, stateGrad, rewardLikelihood.backward(rewardLikelihoodOut, rewardLikelihoodIn, new Tensor[]{grad}, true).getValue().tensors.get(rewardLikelihoodIn[0]));
		}
		
		// Other timesteps:
		// * If next observation dropped:
		//   - Convert gradient of next state sample to gradient on next state prior parameters
		//   - Calculate gradient from next state prior to current state sample
		//   - Add gradient from current observation likelihood if not dropped
		//   - Add gradient from current reward likelihood if modeled
		// * If next observation not dropped:
		//   - Convert gradient of next state sample to gradient on next state posterior parameters
		//   - Add regularization loss based on next state prior
		//   - Calculate gradient from next state posterior parameters to current state sample
		//   - Add gradient from next state prior prior
		//   - Add gradient from current observation likelihood if not dropped
		//   - Add gradient from current reward likelihood if modeled
		for(int t = sequence.size()-2; t >= 0; t--){
			// Convert gradient of s_t+1 to gradient on its parameters
			stateDistributionGrad(sampleParamsGrad, stateGrad, randoms.get(t+1));
			
			// Get prior parameters of s_t+1
			Tensor priorParams = prior.forward(priorIn, priorOut, new Tensor[]{states.get(t), sequence.getAction(t)}).getValue().tensor;
			
			// If dropped, gradient on prior of s_t+1 is gradient on sample parameters of s_t+1
			Tensor priorParamsGrad = sampleParamsGrad;
			
			if(!dropped[t+1]) {
				// Get posterior parameters of s_t+1
				Tensor posteriorParams = posterior.forward(posteriorIn, posteriorOut, new Tensor[]{states.get(t), sequence.getAction(t), sequence.getState(t+1)}).getValue().tensor;
				
				// Add regularization loss based on prior on s_t+1 to gradient on sample parameters of s_t+1
				posteriorRegulLoss += TensorOps.mean(posteriorRegulCriterion.loss(posteriorParams, priorParams));
				TensorOps.add(sampleParamsGrad, sampleParamsGrad, posteriorRegulCriterion.grad(posteriorParams, priorParams));
				
				// Calculate gradient to prior of s_t+1 using regularization loss
				priorParamsGrad = posteriorRegulCriterion.gradTarget(posteriorParams, priorParams);
			}
			
			// Optional prior regularization
			if(config.priorRegularization > 0) {
				priorRegulLoss += config.priorRegularization*TensorOps.mean(priorRegulCriterion.loss(priorParams, referencePriorParams));
				TensorOps.add(priorParamsGrad, priorParamsGrad, config.priorRegularization, priorRegulCriterion.grad(priorParams, referencePriorParams));
			}
			
			// Gradient to s_t is gradient from prior parameters of s_t+1
			stateGrad = prior.backward(priorOut, priorIn, new Tensor[]{priorParamsGrad}, true).getValue().tensors.get(priorIn[0]);
			
			// If o_t+1 not dropped, add gradient to s_t from posterior parameters of s_t+1
			if(!dropped[t+1])
				TensorOps.add(stateGrad, stateGrad, posterior.backward(posteriorOut, posteriorIn, new Tensor[]{sampleParamsGrad}, true).getValue().tensors.get(posteriorIn[0]));
			
			if(!dropped[t]) {
				// If o_t not dropped, add gradient from likelihood of o_t
				Tensor observationReconParams = observationLikelihood.forward(states.get(t));
				Tensor observation = sequence.getState(t);
				
				observationReconLoss += TensorOps.mean(observationReconCriterion.loss(observationReconParams, observation));
				TensorOps.add(stateGrad, stateGrad, observationLikelihood.backward(observationReconCriterion.grad(observationReconParams, observation), true));
			}
			
			if(rewardLikelihood != null) {
				// Optional: reconstruction loss on r_t
				Tensor rewardReconParams = rewardLikelihood.forward(rewardLikelihoodIn, rewardLikelihoodOut, new Tensor[]{states.get(t), sequence.getAction(t)}).getValue().tensor;
				Tensor reward = sequence.getReward(t);
				Tensor terminal = sequence.getTerminal(t);
				terminal.reshape(config.batchSize);
				
				Tensor loss = rewardReconCriterion.loss(rewardReconParams, reward);
				TensorOps.cmul(loss, loss, terminal);
				rewardReconLoss += TensorOps.mean(loss);
				
				Tensor grad = rewardReconCriterion.grad(rewardReconParams, reward);
				rmul(grad, grad, terminal);
				TensorOps.add(stateGrad, stateGrad, rewardLikelihood.backward(rewardLikelihoodOut, rewardLikelihoodIn, new Tensor[]{grad}, true).getValue().tensors.get(rewardLikelihoodIn[0]));
			}
		}
		
		// Additional for first timestep: loss on posterior and prior of s_0
		action.fill(0.0f);
		state.fill(0.0f);
		
		// Convert gradient of s_0 to gradient on its parameters
		stateDistributionGrad(sampleParamsGrad, stateGrad, randoms.get(0));
		
		// Get prior parameters of s_0
		Tensor priorParams = prior.forward(priorIn, priorOut, new Tensor[]{state, action}).getValue().tensor;
		
		// If dropped, gradient on prior of s_0 is gradient on sample parameters of s_0
		Tensor priorParamsGrad = sampleParamsGrad;
		
		// Check if dropped
		if(!dropped[0]) {
			// Get posterior parameters of s_0
			Tensor posteriorParams = posterior.forward(posteriorIn, posteriorOut, new Tensor[]{state, action, sequence.getState(0)}).getValue().tensor;
			
			// Add regularization loss based on prior on s_0 to gradient on sample parameters of s_0
			posteriorRegulLoss += TensorOps.mean(posteriorRegulCriterion.loss(posteriorParams, priorParams));
			TensorOps.add(sampleParamsGrad, sampleParamsGrad, posteriorRegulCriterion.grad(posteriorParams, priorParams));
			
			// Calculate gradient to prior of s_0 using regularization loss
			priorParamsGrad = posteriorRegulCriterion.gradTarget(posteriorParams, priorParams);
		}
		
		// Optional prior regularization
		if(config.priorRegularization > 0) {
			priorRegulLoss += config.priorRegularization*TensorOps.mean(priorRegulCriterion.loss(priorParams, referencePriorParams));
			TensorOps.add(priorParamsGrad, priorParamsGrad, config.priorRegularization, priorRegulCriterion.grad(priorParams, referencePriorParams));
		}
		
		// Calculate gradient of prior of s_0
		prior.backward(priorOut, priorIn, new Tensor[]{priorParamsGrad}, true).getValue();
		
		// Calculate gradient of posterior of s_0
		if(!dropped[0])
			posterior.backward(posteriorOut, posteriorIn, new Tensor[]{sampleParamsGrad}, true).getValue();
		
		// Calculate the deltas
		priorProcessor.calculateDelta(i);
		posteriorProcessor.calculateDelta(i);
		observationLikelihoodProcessor.calculateDelta(i);
		if(rewardLikelihood != null)
			rewardLikelihoodProcessor.calculateDelta(i);
		
		// Update the parameters
		prior.updateParameters();
		posterior.updateParameters();
		observationLikelihood.updateParameters();
		if(rewardLikelihood != null)
			rewardLikelihood.updateParameters();
		
		return new LearnProgress(i, (priorRegulLoss+posteriorRegulLoss+observationReconLoss+rewardReconLoss)/sequence.size);
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
	
	//TODO: add this to TensorOps?
	private static Tensor rmul(Tensor res, Tensor mat, Tensor vec) {
		res = mat.copyInto(res);
		for(int i = 0; i < mat.size(0); i++) {
			Tensor row = mat.select(0, i);
			TensorOps.mul(row, row, vec.get(i));
		}
		return res;
	}
}
