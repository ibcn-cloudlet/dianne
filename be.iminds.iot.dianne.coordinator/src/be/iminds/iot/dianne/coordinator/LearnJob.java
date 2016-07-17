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
package be.iminds.iot.dianne.coordinator;

import java.util.Dictionary;
import java.util.HashMap;
import java.util.Hashtable;
import java.util.Map;
import java.util.UUID;
import java.util.stream.Collectors;

import org.osgi.framework.ServiceRegistration;

import be.iminds.iot.dianne.api.coordinator.Job.EvaluationCategory;
import be.iminds.iot.dianne.api.coordinator.Job.LearnCategory;
import be.iminds.iot.dianne.api.coordinator.Job.Type;
import be.iminds.iot.dianne.api.coordinator.LearnResult;
import be.iminds.iot.dianne.api.nn.eval.Evaluation;
import be.iminds.iot.dianne.api.nn.eval.Evaluator;
import be.iminds.iot.dianne.api.nn.learn.LearnProgress;
import be.iminds.iot.dianne.api.nn.learn.Learner;
import be.iminds.iot.dianne.api.nn.learn.LearnerListener;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkDTO;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkInstanceDTO;

public class LearnJob extends AbstractJob<LearnResult> implements LearnerListener {

	private ServiceRegistration reg;
	
	private long maxIterations = -1;
	
	private LearnResult result = new LearnResult();
	
	private Map<UUID, Learner> learners = new HashMap<>();

	private Evaluator validator = null;
	private Map<UUID, NeuralNetworkInstanceDTO> nnis2 = new HashMap<>();
	private NeuralNetworkInstanceDTO validationNni;
	private float bestValidationError = Float.MAX_VALUE;
	private float miniBatchErrorThreshold = -Float.MAX_VALUE;
	private float validationErrorThreshold = -Float.MAX_VALUE;
	private int errorThresholdWindow = 10;
	
	public LearnJob(DianneCoordinatorImpl coord, 
			NeuralNetworkDTO nn,
			String d,
			Map<String, String> c){
		super(coord, Type.LEARN, nn, d, c);
		
		if(c.containsKey("environment") || coord.isExperiencePool(d)){
			category = LearnCategory.RL;
		} else if(coord.isRecurrent(nn)){
			category = LearnCategory.RNN;
		} else {
			category = LearnCategory.FF;
		}
		
		if(c.containsKey("miniBatchErrorThreshold")){
			miniBatchErrorThreshold = Float.parseFloat(c.get("miniBatchErrorThreshold"));
		}
		
		if(c.containsKey("validationErrorThreshold")){
			validationErrorThreshold = Float.parseFloat(c.get("validationErrorThreshold"));
		}
		
		if(c.containsKey("errorThresholdWindow")){
			errorThresholdWindow = Integer.parseInt(c.get("errorThresholdWindow"));
		}
	}
		
	@Override
	public void execute() throws Exception {
		
		if(coordinator.isExperiencePool(dataset)){
			// DeepQ learner also requires target nni
			nnis2 = new HashMap<>();
			for(UUID target : targets){
				NeuralNetworkInstanceDTO nni = coordinator.platform.deployNeuralNetwork(nn.name, "Dianne Coordinator LearnJob "+jobId, target);
				nnis2.put(target, nni);
			}
		}
		
		// check if we need to do validation - if so get an Evaluator on one of the targets
		if(config.containsKey("validationSet")){
			for(UUID target : targets){
				validator = coordinator.evaluators.get(EvaluationCategory.CRITERION.toString()).get(target);
				if(validator != null){
					validationNni = coordinator.platform.deployNeuralNetwork(nn.name, "Dianne Coordinator LearnJob Validaton NN"+jobId, target);
					break;
				}
			}
		}
		
		if(config.containsKey("maxIterations")){
			maxIterations = Long.parseLong(config.get("maxIterations"));
		}
		
		if(!config.containsKey("tag")){
			config.put("tag", jobId.toString());
		}
		
		Dictionary<String, Object> props = new Hashtable();
		String[] t = targets.stream().map(uuid -> uuid.toString()).collect(Collectors.toList()).toArray(new String[targets.size()]);
		props.put("targets", t);
		props.put("aiolos.unique", true);
		reg = coordinator.context.registerService(LearnerListener.class, this, props);
		
		// set training set
		final Map<String, String> learnConfig = new HashMap<>(config);
		if(config.containsKey("trainSet"))
			learnConfig.put("range", config.get("trainSet"));

		System.out.println("Start Learn Job");
		System.out.println("===============");
		System.out.println("* nn: "+nn.name);
		System.out.println("* dataset: "+dataset);
		System.out.println("* maxIterations: "+maxIterations);
		System.out.println("* miniBatchErrorThreshold: "+(miniBatchErrorThreshold==-Float.MAX_VALUE ? "N\\A" : miniBatchErrorThreshold));
		System.out.println("* validationErrorThreshold: "+(validationErrorThreshold==-Float.MAX_VALUE ? "N\\A" : validationErrorThreshold));
		System.out.println("* errorThresholdWindow: "+errorThresholdWindow);
		System.out.println("---");
		
		// start learning on each learner
		for(UUID target : targets){
			Learner learner = coordinator.learners.get(category.toString()).get(target);
			learners.put(target, learner);
			// if nnis2 is unused, this will give null, but doesnt matter?
			learner.learn(dataset, learnConfig, nnis.get(target), nnis2.get(target));
		}
	}
	
	@Override
	public void onProgress(UUID learnerId, LearnProgress progress) {
		if(deferred.getPromise().isDone()){
			return;
		}
		
		result.progress.add(progress);
		
		// run validation
		Evaluation validation = null;
		if(validator != null){
			Map<String, String> c = new HashMap<>();
			c.put("range", config.get("validationSet"));
			if(config.containsKey("tag")){
				c.put("tag", config.get("tag"));
			} else {
				c.put("tag", jobId.toString());
			}
			if(config.containsKey("criterion")){
				c.put("criterion", config.get("criterion"));
			}
			// TODO use separate (bigger?) batchSize here?
			if(config.containsKey("batchSize")){
				c.put("batchSize", config.get("batchSize"));
			}
			c.put("storeIfBetterThan", ""+bestValidationError);
			
			try {
				validation = validator.eval(dataset, c, validationNni);
				
				if(Float.isNaN(validation.error)){
					validation = null;
					throw new Exception("Validation error became NaN");
				}
				
				if(validation.error < bestValidationError){
					bestValidationError = validation.error;
				}
			} catch(Exception e){
				System.err.println("Error running validation: "+e.getMessage());
			}
			result.validations.add(validation);
		}

		coordinator.sendLearnProgress(this.jobId, progress, validation);

		
		// maxIterations stop condition
		// what in case of multiple learners?!
		boolean stop = maxIterations > 0 && progress.iteration >= maxIterations;
			
		// threshold on delta minibatch error
		// if less 'progress' than this, stop
		if(result.progress.size() > errorThresholdWindow){
			int last = result.progress.size() - 1;
			int prev = last - errorThresholdWindow;
			float deltaMiniBatchError = result.progress.get(prev).error - result.progress.get(last).error;
			if(deltaMiniBatchError < miniBatchErrorThreshold){
				stop = true;
			}
		}
		
		// threshold on validation threshold
		if(result.validations.size() > errorThresholdWindow){
			int last = result.validations.size()-1;
			int prev = last - errorThresholdWindow;
			float deltaValidationError = result.validations.get(prev).error - result.validations.get(last).error;
			if(deltaValidationError < validationErrorThreshold){
				stop = true;
			}
		}
		
		// if stop ... assemble result object and resolve
		if(stop){
			for(Learner learner : learners.values()){
				learner.stop();
			}
		}
	}

	@Override
	public void onException(UUID learnerId, Throwable e) {
		if(deferred.getPromise().isDone()){
			return;
		}
		done(e);
	}

	@Override
	public void onFinish(UUID learnerId, LearnProgress p) {
		result.progress.add(p);
		done(result);
	}

	@Override
	public void cleanup() {
		if(reg!=null)
			reg.unregister();

		for(Learner learner : learners.values()){
			learner.stop();
		}
		
		// these are used in case of deep q learning
		for(NeuralNetworkInstanceDTO nni : nnis2.values()){
			coordinator.platform.undeployNeuralNetwork(nni);
		}
		
		if(validationNni != null)
			coordinator.platform.undeployNeuralNetwork(validationNni);
	}

	@Override
	public LearnResult getProgress() {
		return result;
	}


	@Override
	public void stop() throws Exception{
		if(started > 0){
			done(result);
		} else {
			done(new Exception("Job "+this.jobId+" cancelled."));
		}
	}

}
