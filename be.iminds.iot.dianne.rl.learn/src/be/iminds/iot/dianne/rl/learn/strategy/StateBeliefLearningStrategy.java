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
 * This strategy tries to train an unsupervized state representation from observations by
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
	
	protected NeuralNetwork encoder;
	protected NeuralNetwork decoder;
	protected NeuralNetwork predictor;

	protected UUID[] encoderIn;
	protected UUID[] encoderOut;
	
	protected UUID[] predictorIn;
	protected UUID[] predictorOut;
	
	protected Criterion reconCriterion;
	protected Criterion regulCriterion;
	
	protected GradientProcessor encoderGradients;
	protected GradientProcessor decoderGradients;
	protected GradientProcessor predictorGradients;
	
	protected Tensor action;
	protected Tensor state;
	protected Tensor random;
	
	protected Tensor fixedPrior;
	protected Tensor prior;
	protected Tensor posterior;
	
	protected Tensor stateDistributionGrad;
	
	protected List<Tensor> states = new ArrayList<>();
	
	@Override
	public void setup(Map<String, String> config, Dataset dataset, NeuralNetwork... nns) throws Exception {
		if(!(dataset instanceof ExperiencePool))
			throw new RuntimeException("Dataset is no experience pool");
		
		this.pool = (ExperiencePool) dataset;
		
		if(nns.length != 3)
			throw new RuntimeException("Invalid number of NN instances provided: "+nns.length+" (expected 3)");
			
		this.encoder = nns[0];
		this.decoder = nns[1];
		this.predictor = nns[2];
		
		this.encoderIn = encoder.getModuleIds("State","Action","Observation");
		this.encoderOut = new UUID[]{encoder.getOutput().getId()};
		
		this.predictorIn = predictor.getModuleIds("State","Action");
		this.predictorOut = new UUID[]{predictor.getOutput().getId()};
		
		this.config = DianneConfigHandler.getConfig(config, StateBeliefConfig.class);
		this.sampling = SequenceSamplingFactory.createSamplingStrategy(this.config.sampling, this.pool, config);
		// always sample from start index
		indices = new int[this.config.batchSize];
		for(int i=0;i<this.config.batchSize;i++){
			indices[i]=0;
		}
		
		// for now batchSize fixed 1
		this.reconCriterion = CriterionFactory.createCriterion(this.config.criterion, config);
		this.regulCriterion = CriterionFactory.createCriterion(CriterionConfig.GKL, config); 
		
		this.encoderGradients = ProcessorFactory.createGradientProcessor(this.config.method, encoder, config);
		this.decoderGradients = ProcessorFactory.createGradientProcessor(this.config.method, decoder, config);
		this.predictorGradients = ProcessorFactory.createGradientProcessor(this.config.method, predictor, config);

		this.action = new Tensor(this.config.batchSize, pool.actionDims());
		this.state = new Tensor(this.config.batchSize, this.config.stateSize);
		this.random = new Tensor(this.config.batchSize, this.config.stateSize);

		this.stateDistributionGrad = new Tensor(this.config.batchSize, 2*this.config.stateSize);
		this.prior = new Tensor(this.config.batchSize, 2*this.config.stateSize);
		this.fixedPrior = new Tensor(this.config.batchSize, 2*this.config.stateSize);
		this.fixedPrior.narrow(1, 0, this.config.stateSize).fill(0.0f);
		this.fixedPrior.narrow(1, this.config.stateSize, this.config.stateSize).fill(1.0f);
		
		System.out.println("Start learning...");
	}

	@Override
	public LearnProgress processIteration(long i) throws Exception {
		// reset 
		encoder.zeroDeltaParameters();
		decoder.zeroDeltaParameters();
		predictor.zeroDeltaParameters();

		// fetch sequence
		int[] seq = sampling.sequence(config.batchSize);
		sequence = pool.getBatchedSequence(sequence, seq, indices , config.sequenceLength);
		
		// start action/state
		action.fill(0.0f);
		state.fill(0.0f);

		// calculate intermediate state estimates
		for(int k=0;k<sequence.size();k++){
			
			Tensor observation = sequence.getState(k);
			posterior = encoder.forward(encoderIn, encoderOut, new Tensor[]{state, action, observation}).getValue().tensor;
			sampleState(state, posterior);

			storeState(state, k);
			
			action = sequence.getAction(k);
		}
		
		// now go from last to first and calculate:
		
		float reconLoss = 0;
		float regulLoss = 0;
		
		stateDistributionGrad.fill(0.0f);
		
		for(int k=sequence.size()-1;k>=0;k--){
		
			// reconstruction error + gradient
			Tensor reconstruction = decoder.forward(states.get(k));
			float recl = TensorOps.mean(reconCriterion.loss(reconstruction, sequence.getState(k)));
			reconLoss += recl;
			
			Tensor reconstructionGrad = decoder.backward(reconCriterion.grad(reconstruction, sequence.getState(k)));
			decoder.accGradParameters();

			accStateDistributionGrad(stateDistributionGrad, reconstructionGrad);
			
			float regl = 0;
			if(!config.fixedPrior && k>0){
				// calculate prior
				prior = predictor.forward(predictorIn, predictorOut, new Tensor[]{states.get(k-1), sequence.getAction(k-1)}).getValue().tensor;
				
				// re calculate posterior
				posterior = encoder.forward(encoderIn, encoderOut, new Tensor[]{states.get(k-1), sequence.getAction(k-1), sequence.getState(k)}).getValue().tensor;

				// KL divergence
				regl = TensorOps.mean(regulCriterion.loss(posterior, prior));
				
				// backward encoder 
				Tensor posteriorGrad = regulCriterion.grad(posterior, prior);
				TensorOps.add(stateDistributionGrad, stateDistributionGrad, posteriorGrad);
				Tensor stateGrad = encoder.backward(encoderOut, encoderIn, new Tensor[]{stateDistributionGrad}).getValue().tensors.get(encoderIn[0]);
				stateDistributionGrad.fill(0.0f);
				accStateDistributionGrad(stateDistributionGrad, stateGrad);
				encoder.accGradParameters();
				
				// backward prior
				Tensor priorGrad = regulCriterion.gradTarget(posterior, prior);
				
				// also add KL grad wrt fixed prior N(0,1)
				TensorOps.add(priorGrad, priorGrad, regulCriterion.grad(prior, fixedPrior));
				
				predictor.backward(predictorOut, predictorIn, new Tensor[]{priorGrad}).getValue();
				predictor.accGradParameters();
				
			} else {
				// special case for k == 0 : use N(0,1) as prior
				
				state.fill(0.0f);
				action.fill(0.0f);
				// re calculate posterior
				posterior = encoder.forward(encoderIn, encoderOut, new Tensor[]{state, action, sequence.getState(k)}).getValue().tensor;

				// kl divergence
				regl = TensorOps.mean(regulCriterion.loss(posterior, fixedPrior));
				
				// backward encoder 
				Tensor posteriorGrad = regulCriterion.grad(posterior, fixedPrior);
				TensorOps.add(stateDistributionGrad, stateDistributionGrad, posteriorGrad);
				Tensor stateGrad = encoder.backward(encoderOut, encoderIn, new Tensor[]{stateDistributionGrad}).getValue().tensors.get(encoderIn[0]);
				stateDistributionGrad.fill(0.0f);
				accStateDistributionGrad(stateDistributionGrad, stateGrad);
				encoder.accGradParameters();
			}
			regulLoss += regl;
			
			System.out.println("LOSS Reconstruction:\t"+recl+"\tRegularization:\t"+regl);

		}

		encoderGradients.calculateDelta(i);
		decoderGradients.calculateDelta(i);
		if(!config.fixedPrior && config.sequenceLength > 1)
			predictorGradients.calculateDelta(i);
		
		encoder.updateParameters();
		decoder.updateParameters();
		if(!config.fixedPrior && config.sequenceLength > 1)
			predictor.updateParameters();
		
		return new LearnProgress(i, (reconLoss+regulLoss)/sequence.size);
	}
	
	private void storeState(Tensor state, int index){
		while(index>=states.size()){
			states.add(new Tensor());
		}
		state.copyInto(states.get(index));
	}
	
	private void sampleState(Tensor state, Tensor stateDistribution) {
		// latentParam => latent
		Tensor means = stateDistribution.narrow(1, 0, config.stateSize);
		Tensor stdevs = stateDistribution.narrow(1, config.stateSize, config.stateSize);
		
		random.randn();
		
		TensorOps.cmul(state, random, stdevs);
		TensorOps.add(state, state, means);
	}
	
	private void accStateDistributionGrad(Tensor stateDistributionGrad, Tensor stateGrad) {
		// latentGrad => latentParamsGrad
		Tensor gradMeans = stateDistributionGrad.narrow(1, 0, config.stateSize);
		Tensor gradStdevs = stateDistributionGrad.narrow(1, config.stateSize, config.stateSize);
		
		TensorOps.add(gradMeans, gradMeans, stateGrad);
		TensorOps.addcmul(gradStdevs, gradStdevs, 1, stateGrad, random);
	}
}
