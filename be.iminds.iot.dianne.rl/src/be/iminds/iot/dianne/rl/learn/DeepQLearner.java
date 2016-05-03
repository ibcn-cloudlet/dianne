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
package be.iminds.iot.dianne.rl.learn;

import java.util.EnumSet;
import java.util.Map;
import java.util.stream.Collectors;

import org.osgi.service.component.annotations.Component;

import be.iminds.iot.dianne.api.nn.NeuralNetwork;
import be.iminds.iot.dianne.api.nn.learn.Learner;
import be.iminds.iot.dianne.api.nn.module.Module.Mode;
import be.iminds.iot.dianne.api.nn.module.Trainable;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkInstanceDTO;
import be.iminds.iot.dianne.api.rl.dataset.ExperiencePool;
import be.iminds.iot.dianne.api.rl.dataset.ExperiencePoolSample;
import be.iminds.iot.dianne.api.rl.learn.QLearnProgress;
import be.iminds.iot.dianne.nn.learn.AbstractLearner;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;

@Component(service=Learner.class, 
property={"aiolos.unique=true",
		  "dianne.learner.category=RL"})
public class DeepQLearner extends AbstractLearner {

	private final String[] logLabels = new String[]{"Q", "Target Q", "Error"};
	
	private NeuralNetwork target;
	
	private ExperiencePool pool; 

	private int targetInterval = 1000;
	private int storeInterval = 0;
	private String storeTag = "store";
	private int minSamples = 10000;
	
	private int batchSize = 10;
	private float discount = 0.99f;
	
	private float q;

	protected void loadConfig(Map<String, String> config){
		super.loadConfig(config);
		
		if (config.containsKey("targetInterval"))
			targetInterval = Integer.parseInt(config.get("targetInterval"));
		System.out.println("* targetInterval = "+targetInterval);

		if (config.containsKey("storeInterval"))
			storeInterval = Integer.parseInt(config.get("storeInterval"));
		System.out.println("* storeInterval = "+storeInterval);

		if (config.containsKey("storeTag"))
			storeTag = config.get("storeTag");
		System.out.println("* storeTag = "+storeTag);
		
		if (config.containsKey("minSamples"))
			minSamples = Integer.parseInt(config.get("minSamples"));
		System.out.println("* minSamples = "+minSamples);
	
		if(config.get("batchSize")!=null){
			batchSize = Integer.parseInt(config.get("batchSize"));
		}
		System.out.println("* batchSize = "+batchSize);

		if(config.containsKey("discount"))
			discount = Float.parseFloat(config.get("discount"));
		System.out.println("* discount = "+discount);

	}

	
	protected void loadNNs(NeuralNetworkInstanceDTO...nni) throws Exception {
		if(nni.length<2){
			throw new Exception("Deep Q Learner requires 2 neural network instances");
		}
		super.loadNNs(nni);
		
		target = dianne.getNeuralNetwork(nni[1]).getValue();
		target.getModules().values().stream().forEach(m -> m.setMode(EnumSet.of(Mode.BLOCKING)));
	}
	
	protected void loadDataset(String d){
		super.loadDataset(d);
		
		if(dataset instanceof ExperiencePool){
			pool = (ExperiencePool) dataset;
		} else {
			throw new RuntimeException("Dataset is no experience pool");
		}
	}

	
	protected void preprocess(){
		super.preprocess();
		
		// wait until pool has some samples
		if(pool.size() < minSamples){
			System.out.println("Experience pool has too few samples, waiting a bit to start learning...");
			while(pool.size() < minSamples){
				try {
					Thread.sleep(5000);
				} catch (InterruptedException e) {
				}
			}
		}
		System.out.println("Start learning...");
	}
	
	public QLearnProgress getProgress() {
		if(!learning || i==0)
			return null;
		return new QLearnProgress(i, error, q);
	}
	
	protected float process(long i){
		nn.getTrainables().values().stream().forEach(Trainable::zeroDeltaParameters);

		float err = 0;
			
		for(int k=0;k<batchSize;k++){
			// new sample
			int index = sampling.next();
			
			ExperiencePoolSample sample = pool.getSample(index);
			
			Tensor in = sample.input;

			// forward
			Tensor out = nn.forward(in, ""+index);
			
			// evaluate criterion
			Tensor action = sample.action;
			float reward = sample.reward;
			Tensor nextState = sample.nextState;
			
			float targetQ = 0;
			
			if(nextState==null){
				// terminal state
				targetQ = reward;
			} else {
				Tensor nextQ = target.forward(nextState, ""+index);
				targetQ = reward + discount * TensorOps.max(nextQ);
			}
			
			Tensor targetOut = out.copyInto(null);
			targetOut.set(targetQ, TensorOps.argmax(action));
			
			float Q_sa = out.get(TensorOps.argmax(action));
			if(i==0){
				q = Q_sa;
			} else {
				q = (1 - alpha) * q + alpha * Q_sa;
			}
		
			Tensor e = criterion.error(out, targetOut);
			err += e.get(0);
			
			if(logger!=null){
				logger.log("LEARN", logLabels, Q_sa, targetQ, e.get(0));
			}
			
			Tensor gradOut = criterion.grad(out, targetOut);
			
			// backward
			Tensor gradIn = nn.backward(gradOut, ""+index);
			
			// acc gradParameters
			nn.getTrainables().values().stream().forEach(m -> m.accGradParameters());
		}
		
		if(Double.isInfinite(err) || Double.isNaN(err)){
			System.out.println(i+" ERROR IS "+err);
		}
			
		gradientProcessor.calculateDelta(i);

		nn.getTrainables().values().stream().forEach(Trainable::updateParameters);
		
		return err/batchSize;
	}
	
	
	protected void initializeParameters(){
		if(clean){
			// reset parameters
			resetParameters();
		} else {
			// load previous parameters
			loadParameters(nn, target);
		}
	}
	
	protected void publishParameters(long i){
		if (targetInterval > 0 && i % targetInterval == 0) {
			nn.storeDeltaParameters(previousParameters, tag);
			loadParameters(nn, target);
			// also store these parameters tagged with batch number
			nn.storeParameters(""+i);
		} else if(syncInterval > 0 && i % syncInterval == 0){
			nn.storeDeltaParameters(previousParameters, tag);
			loadParameters(nn);
		}
		
		if(storeInterval > 0 && i % storeInterval == 0){
			nn.storeParameters(storeTag, ""+i);
		}
	}

	private void loadParameters(NeuralNetwork... nns) {
		try {
			for(NeuralNetwork nn : nns){
				previousParameters = nn.loadParameters(tag);
			}
		} catch (Exception ex) {
			resetParameters();
		}
	}
	
	private void resetParameters(){
		nn.randomizeParameters();
		
		// store those parameters
		nn.storeParameters(tag);
		
		// copy to target
		target.setParameters(nn.getParameters());
		
		// copy to previousParameters
		previousParameters =  nn.getParameters().entrySet().stream().collect(Collectors.toMap(e -> e.getKey(), e -> e.getValue().copyInto(null)));
	}

}
