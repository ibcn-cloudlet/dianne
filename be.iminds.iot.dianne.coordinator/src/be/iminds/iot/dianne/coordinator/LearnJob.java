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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Dictionary;
import java.util.HashMap;
import java.util.Hashtable;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.stream.Collectors;

import org.osgi.framework.ServiceRegistration;

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

	private ServiceRegistration<LearnerListener> reg;
	
	private long maxIterations = -1;
	
	private LearnResult result = new LearnResult();
	
	// in case of multiple learners, one is assigned "master" 
	// running (optional) validation and checking stop condition
	private UUID master = null;
	private Map<UUID, Learner> learners = new HashMap<>();

	private Evaluator validator = null;
	private NeuralNetworkInstanceDTO[] validationNns;
	private int validationInterval = 1000;
	private float bestValidationLoss = Float.MAX_VALUE;
	
	
	private float minibatchLossThreshold = -Float.MAX_VALUE;
	private float validationLossThreshold = -Float.MAX_VALUE;
	private int thresholdWindow = 10;
	
	public LearnJob(DianneCoordinatorImpl coord, 
			String dataset,
			Map<String, String> config,
			NeuralNetworkDTO[] nns){
		super(coord, Type.LEARN, dataset, config, nns);
		
		if(config.containsKey("strategy") && config.get("strategy").equals("FeedForwardLearningStrategy")){
			// set FF category when explicitly using FF strategy
			category = LearnCategory.FF;
		} else if(config.containsKey("environment")
				|| coord.datasets.isExperiencePool(dataset)){
			category = LearnCategory.RL;
		} else if(coord.isRecurrent(nns[0])){
			category = LearnCategory.RNN;
		} else {
			category = LearnCategory.FF;
		}

		if(config.containsKey("validationInterval")){
			validationInterval = Integer.parseInt(config.get("validationInterval"));
		}
		
		
		if(config.containsKey("minibatchLossThreshold")){
			minibatchLossThreshold = Float.parseFloat(config.get("minibatchLossThreshold"));
		}
		
		if(config.containsKey("validationLossThreshold")){
			validationLossThreshold = Float.parseFloat(config.get("validationLossThreshold"));
		}
		
		if(config.containsKey("thresholdWindow")){
			thresholdWindow = Integer.parseInt(config.get("thresholdWindow"));
		}
	}
		
	@Override
	public void execute() throws JobFailedException {
		
		master = targets.get(0);
		if(config.containsKey("validationSet")){
			for(UUID target : targets){
				Evaluator v = coordinator.evaluators.get(target);
				if(v != null){
					try {
						validator = v;
						// validator becomes master
						master = target;
						validationNns = new NeuralNetworkInstanceDTO[nns.length];
						for(int i=0;i<nns.length;i++){
							validationNns[i] = coordinator.platform.deployNeuralNetwork(nns[i].name, "Dianne Coordinator LearnJob Validaton NN"+jobId, super.config, target);
						}
						break;
					} catch(Throwable c){
						throw new JobFailedException(target, this.jobId, "Failed initialize validation: "+c.getMessage(), c);
					}
				}
			}
		}
		
		if(config.containsKey("maxIterations")){
			maxIterations = Long.parseLong(config.get("maxIterations"));
		}
		
		if(!config.containsKey("tag")){
			config.put("tag", jobId.toString());
		}
		
		Dictionary<String, Object> props = new Hashtable<>();
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
		System.out.println("* nn: "+Arrays.toString(nnNames));
		System.out.println("* dataset: "+dataset);
		System.out.println("* maxIterations: "+maxIterations);
		System.out.println("* minibatchLossThreshold: "+(minibatchLossThreshold==-Float.MAX_VALUE ? "N\\A" : minibatchLossThreshold));
		System.out.println("* validationLossThreshold: "+(validationLossThreshold==-Float.MAX_VALUE ? "N\\A" : validationLossThreshold));
		System.out.println("* thresholdWindow: "+thresholdWindow);
		System.out.println("---");
		
		// start learning on each learner
		for(UUID target : targets){
			try {
				Learner learner = coordinator.learners.get(target);
				learners.put(target, learner);
				learner.learn(dataset, learnConfig, nnis.get(target));
			} catch(Throwable c){
				throw new JobFailedException(target, this.jobId, "Failed to start learner: "+c.getMessage(), c);
			}
		}
	}
	
	@Override
	public void onProgress(UUID learnerId, LearnProgress progress) {
		if(deferred.getPromise().isDone()){
			return;
		}
		
		if(result.progress.get(learnerId)==null){
			List<LearnProgress> p = new ArrayList<>();
			result.progress.put(learnerId, p);
		}
		result.progress.get(learnerId).add(progress);

		// run validations / publish progress and check stop condition on
		// elected 'master' node
		if(learnerId.equals(master)){
			// run validation
			Evaluation validation = null;
			if(validator != null
					&& progress.iteration % validationInterval == 0){
				Map<String, String> c = new HashMap<>();
				c.put("range", config.get("validationSet"));
				if(config.containsKey("tag")){
					c.put("tag", config.get("tag"));
				} else {
					c.put("tag", jobId.toString());
				}
				if(config.containsKey("validationStrategy")){
					c.put("strategy", config.get("validationStrategy"));
				} else {
					c.put("strategy", "CriterionEvaluationStrategy");
				}
				if(config.containsKey("criterion")){
					c.put("criterion", config.get("criterion"));
				}
				// TODO use separate (bigger?) batchSize here?
				if(config.containsKey("batchSize")){
					c.put("batchSize", config.get("batchSize"));
				}
				c.put("storeIfBetterThan", ""+bestValidationLoss);
				
				try {
					validation = validator.eval(dataset, c, validationNns);
					
					if(Float.isNaN(validation.metric)){
						validation = null;
						throw new Exception("Validation loss became NaN");
					}
					
					if(validation.metric < bestValidationLoss){
						bestValidationLoss = validation.metric;
					}
				} catch(Exception e){
					System.err.println("Error running validation: "+e.getMessage());
					onException(learnerId, e);
					return;
				}
				result.validations.put(progress.iteration, validation);
			}

			// TODO how frequently send out the progress
			coordinator.sendLearnProgress(this.jobId, progress, validation);
			
			// maxIterations stop condition
			// what in case of multiple learners?!
			boolean stop = maxIterations > 0 && progress.iteration >= maxIterations;
				
			// threshold on delta minibatch error
			// if less 'progress' than this, stop
			if(result.progress.size() > thresholdWindow){
				int last = result.progress.size() - 1;
				int prev = last - thresholdWindow;
				float deltaMinibatchLoss = result.progress.get(learnerId).get(prev).minibatchLoss - result.progress.get(learnerId).get(last).minibatchLoss;
				if(deltaMinibatchLoss < minibatchLossThreshold){
					stop = true;
				}
			}
			
			// threshold on validation threshold
			Evaluation last = result.validations.get(progress.iteration);
			Evaluation prev = result.validations.get(progress.iteration - thresholdWindow*validationInterval);
			if(last != null & prev != null){
				float deltaValidationLoss = prev.metric - last.metric;
				if(deltaValidationLoss < validationLossThreshold){
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
	}

	@Override
	public void onException(UUID learnerId, Throwable e) {
		if(deferred.getPromise().isDone()){
			return;
		}
		done(new JobFailedException(learnerId, this.jobId, "Learner failed: "+e.getMessage(), e));
	}

	@Override
	public void onFinish(UUID learnerId) {
		if(deferred.getPromise().isDone()){
			return;
		}
		done(result);
	}

	@Override
	public void cleanup() {
		if(reg!=null)
			reg.unregister();

		// just to be sure all are stopped in case of errors
		for(Learner learner : learners.values()){
			try {
				learner.stop();
			} catch(Exception e){}
		}
		
		if(validationNns != null){
			for(NeuralNetworkInstanceDTO nn : validationNns)
				coordinator.platform.undeployNeuralNetwork(nn);
		}
	}

	@Override
	public LearnResult getProgress() {
		return result;
	}


	@Override
	public void stop() throws Exception{
		if(started > 0){
			// just stop the learners to get intermediate results
			for(Learner learner : learners.values()){
				try {
					learner.stop();
				} catch(Exception e){}
			}		
		} else {
			done(new JobFailedException(null, this.jobId, "Job "+this.jobId+" cancelled.", null));
		}
	}

}
