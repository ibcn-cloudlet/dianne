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
import java.util.UUID;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.nn.NeuralNetwork;
import be.iminds.iot.dianne.api.nn.learn.Criterion;
import be.iminds.iot.dianne.api.nn.learn.GradientProcessor;
import be.iminds.iot.dianne.api.nn.learn.LearnProgress;
import be.iminds.iot.dianne.api.nn.learn.LearningStrategy;
import be.iminds.iot.dianne.api.nn.learn.SamplingStrategy;
import be.iminds.iot.dianne.api.rl.dataset.ExperiencePool;
import be.iminds.iot.dianne.api.rl.dataset.ExperiencePoolBatch;
import be.iminds.iot.dianne.nn.learn.criterion.CriterionFactory;
import be.iminds.iot.dianne.nn.learn.processors.ProcessorFactory;
import be.iminds.iot.dianne.nn.learn.sampling.SamplingFactory;
import be.iminds.iot.dianne.nn.util.DianneConfigHandler;
import be.iminds.iot.dianne.rl.learn.strategy.config.StateBeliefConfig;
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
	protected SamplingStrategy sampling;
	
	protected ExperiencePoolBatch batch;
	
	protected NeuralNetwork encoder;
	protected NeuralNetwork decoder;
	protected NeuralNetwork predictor;

	protected UUID[] predictorIns;
	protected UUID[] predictorOut;
	
	protected Criterion decoderCriterion;
	protected Criterion predictorCriterion;
	protected GradientProcessor encoderGradients;
	protected GradientProcessor decoderGradients;
	protected GradientProcessor predictorGradients;

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
		
		this.predictorIns = new UUID[]{predictor.getModuleId("Input"), predictor.getModuleId("Action")};
		this.predictorOut = new UUID[]{predictor.getOutput().getId()};
		
		this.config = DianneConfigHandler.getConfig(config, StateBeliefConfig.class);
		this.sampling = SamplingFactory.createSamplingStrategy(this.config.sampling, dataset, config);
		
		this.decoderCriterion = CriterionFactory.createCriterion(this.config.criterion, config);
		this.predictorCriterion = CriterionFactory.createCriterion(this.config.criterion, config);

		this.encoderGradients = ProcessorFactory.createGradientProcessor(this.config.method, encoder, config);
		this.decoderGradients = ProcessorFactory.createGradientProcessor(this.config.method, decoder, config);
		this.predictorGradients = ProcessorFactory.createGradientProcessor(this.config.method, predictor, config);

		System.out.println("Start learning...");
	}

	@Override
	public LearnProgress processIteration(long i) throws Exception {
		// Reset the deltas
		encoder.zeroDeltaParameters();
		decoder.zeroDeltaParameters();
		predictor.zeroDeltaParameters();
		
		batch = pool.getBatch(batch, sampling.next(config.batchSize));
		
		Tensor state = encoder.forward(batch.getState());
		Tensor action = batch.getAction();
		
		// calc reconstruction and prediction
		Tensor reconstruction  = decoder.forward(state);
		Tensor prediction = predictor.forward(predictorIns, predictorOut, new Tensor[]{state, action}).getValue().tensor;
		
		// calc loss
		float reconstructionLoss = TensorOps.mean(decoderCriterion.loss(reconstruction, batch.getState()));
		float predictionLoss = TensorOps.mean(predictorCriterion.loss(prediction, batch.getNextState()));
		float loss = reconstructionLoss+predictionLoss;
		
		// calc gradients
		Tensor reconstructionGrad = decoderCriterion.grad(reconstruction, batch.getState());
		Tensor predictionGrad = predictorCriterion.grad(prediction, batch.getNextState());

		
		// backward
		Tensor encoderGrad = decoder.backward(reconstructionGrad);
		TensorOps.add(encoderGrad, encoderGrad, predictor.backward(predictorOut, new UUID[]{predictorIns[0]}, new Tensor[]{predictionGrad}).getValue().tensor);
		encoder.backward(encoderGrad);
		
		// acc grad
		decoder.accGradParameters();
		predictor.accGradParameters();
		encoder.accGradParameters();
		
		decoderGradients.calculateDelta(i);
		predictorGradients.calculateDelta(i);
		encoderGradients.calculateDelta(i);

		decoder.updateParameters();
		predictor.updateParameters();
		encoder.updateParameters();
		
		return new LearnProgress(i, loss);
	}

}
