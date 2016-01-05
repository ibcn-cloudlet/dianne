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

import java.util.Collection;
import java.util.Dictionary;
import java.util.HashMap;
import java.util.Hashtable;
import java.util.Map;
import java.util.Queue;
import java.util.UUID;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingQueue;

import org.osgi.framework.BundleContext;
import org.osgi.framework.ServiceRegistration;
import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;
import org.osgi.util.promise.Deferred;
import org.osgi.util.promise.Promise;

import be.iminds.iot.dianne.api.coordinator.DianneCoordinator;
import be.iminds.iot.dianne.api.coordinator.LearnResult;
import be.iminds.iot.dianne.api.nn.eval.Evaluation;
import be.iminds.iot.dianne.api.nn.eval.Evaluator;
import be.iminds.iot.dianne.api.nn.learn.LearnProgress;
import be.iminds.iot.dianne.api.nn.learn.Learner;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkDTO;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkInstanceDTO;
import be.iminds.iot.dianne.api.nn.platform.DiannePlatform;
import be.iminds.iot.dianne.api.repository.DianneRepository;
import be.iminds.iot.dianne.api.repository.RepositoryListener;

@Component
public class DianneCoordinatorImpl implements DianneCoordinator {

	private BundleContext context;

	private DiannePlatform platform;
	private DianneRepository repository;
	// TODO for now only one learner and evaluator
	private Learner learner;
	private Evaluator evaluator;
	//private Map<UUID, Learner> learners = Collections.synchronizedMap(new HashMap<>());
	//private Map<UUID, Evaluator> evaluators = Collections.synchronizedMap(new HashMap<>());
	
	private Queue<LearnJob> queue = new LinkedBlockingQueue<>();

	
	private ExecutorService pool = Executors.newCachedThreadPool();
	
	@Override
	public Promise<LearnResult> learn(NeuralNetworkDTO nn, String dataset, Map<String, String> config) {
		repository.storeNeuralNetwork(nn);
		
		LearnJob job = new LearnJob(nn, config, dataset);
		queue.add(job);
		next();
		
		return job.getPromise();
	}

	@Override
	public Promise<LearnResult> learn(String nnName, String dataset, Map<String, String> config) {
		return learn(repository.loadNeuralNetwork(nnName), dataset, config);
	}
	
	@Override
	public Promise<Evaluation> eval(NeuralNetworkDTO nn, String dataset, Map<String, String> config) {
		// TODO evaluate time on other/multiple platforms?
		repository.storeNeuralNetwork(nn);
		
		EvaluationJob job = new EvaluationJob(nn, config, dataset);
		job.start(evaluator);
		
		return job.getPromise();
	}

	@Override
	public Promise<Evaluation> eval(String nnName, String dataset, Map<String, String> config) {
		return eval(repository.loadNeuralNetwork(nnName), dataset, config);
	}
	
	
	// try to do next job
	private void next(){
		// TODO one should keep a separate list of available learners
		// when calling start the learner is not yet busy...
		if(!learner.isBusy()){
			LearnJob job = queue.poll();
			if(job!=null){
				job.start(learner);
			}
		}
	}
	
	private class LearnJob implements RepositoryListener, Runnable {
		
		private Deferred<LearnResult> deferred = new Deferred<>();
		
		private NeuralNetworkDTO nn;
		private Map<String, String> config;
		private String dataset;
		
		private Learner learner;
		private NeuralNetworkInstanceDTO nni;
		
		private ServiceRegistration<RepositoryListener> reg;
		
		private boolean done = false;
		
		private long maxIterations = Long.MAX_VALUE;
		
		public LearnJob(NeuralNetworkDTO nn, Map<String, String> config, String dataset){
			this.nn = nn;
			this.config = config;
			this.dataset = dataset;
			
			if(config.containsKey("maxIterations")){
				maxIterations = Long.parseLong(config.get("maxIterations"));
			}
		}
		
		Promise<LearnResult> getPromise(){
			return deferred.getPromise();
		}
		
		public void start(Learner learner){
			this.learner = learner;
			pool.execute(this);
		}
		
		public void run(){
			System.out.println("Start Learn Job");
			System.out.println("===============");
			System.out.println("* nn: "+nn.name);
			System.out.println("* dataset: "+dataset);
			System.out.println("* maxIterations: "+maxIterations);
			System.out.println("---");

	
			// do the actual Learning
			try {
				// deploy nn
				nni = platform.deployNeuralNetwork(nn.name, "Dianne Coordinator LearnJob", learner.getLearnerId());
				
				// register RepositoryListener
				Dictionary<String, Object> props = new Hashtable();
				props.put("targets", new String[]{":"+nni.id.toString()});
				props.put("aiolos.unique", true);
				reg = context.registerService(RepositoryListener.class, this, props);
				
				// TODO provide dataset?
				
				// set trainging set
				Map<String, String> learnConfig = new HashMap<>(config);
				learnConfig.put("range", config.get("trainingSet"));
				
				// start learning
				learner.learn(dataset, learnConfig, nni);
				
				// TODO timeout?
			
			} catch(Exception e){
				System.out.println("Job failed");
				e.printStackTrace();
				
				deferred.fail(e);
				done();
			}
		}

		@Override
		public void onParametersUpdate(UUID nnId, Collection<UUID> moduleIds,
				String... tag) {
			if(done){
				return;
			}
			
			// check stop conditition
			LearnProgress progress = learner.getProgress();
			
			// maxIterations stop condition 
			boolean stop = progress.iteration >= maxIterations;
			
			// TODO other stop conditions
			// - check error rate evolution (on train and/or validationSet)
			// - stop at a certain error rate
			// - stop after certain time
			// ...
			
			// if stop ... assemble result object and resolve
			if(stop){
				learner.stop();
				
				try {
					LearnResult result = new LearnResult(progress.error, progress.iteration);
					deferred.resolve(result);
				} catch(Exception e){
					deferred.fail(e);
				} finally {
					done();
				}
			}
		}
		
		private void done(){
			done = true;
			
			if(reg!=null)
				reg.unregister();
			
			if(nni!=null)
				platform.undeployNeuralNetwork(nni);
			
			next();
		}
	}

	
	private class EvaluationJob implements Runnable {

		private Deferred<Evaluation> deferred = new Deferred<>();
		
		private NeuralNetworkDTO nn;
		private Map<String, String> config;
		private String dataset;
		
		private Evaluator evaluator;
		private NeuralNetworkInstanceDTO nni;
		
		public EvaluationJob(NeuralNetworkDTO nn, Map<String, String> config, String dataset){
			this.nn = nn;
			this.config = config;
			this.dataset = dataset;
		}
		
		public void start(Evaluator evaluator){
			this.evaluator = evaluator;
			pool.execute(this);
		}
		
		@Override
		public void run() {
			try {
				// deploy nn
				nni = platform.deployNeuralNetwork(nn.name, "Dianne Coordinator EvaluationJob", evaluator.getEvaluatorId());
				
				Map<String, String> evalConfig = new HashMap<>(config);
				if(!evalConfig.containsKey("range")){
					evalConfig.put("range", config.get("testSet"));
				}
				
				Evaluation eval = evaluator.eval(nni, dataset, evalConfig);
				deferred.resolve(eval);
				
			} catch(Exception e){
				deferred.fail(e);
			} finally {
				if(nni!=null){
					platform.undeployNeuralNetwork(nni);
				}
			}
		}
		
		public Promise<Evaluation> getPromise(){
			return deferred.getPromise();
		}
		
	}
	
	private void parseRange(Map<String, String> config, String set){
		String range = config.get(set);
		if(range!=null){
			try {
				String[] split = range.split(":");
				int startIndex = Integer.parseInt(split[0]);
				int endIndex = Integer.parseInt(split[1]);
				
				config.put("startIndex", ""+startIndex);
				config.put("endIndex", ""+endIndex);
			} catch(Exception e){
				System.out.println(set+" wrongly specified, should be startIndex:endIndex");
			}
		}
	}
	
	@Activate
	void activate(BundleContext context){
		this.context = context;
	}
	
	@Reference
	void setDiannePlatform(DiannePlatform platform){
		this.platform = platform;
	}
	
	@Reference
	void setDianneRepository(DianneRepository repository){
		this.repository = repository;
	}
	
	@Reference
	void setLearner(Learner learner){
		this.learner = learner;
	}
	
	@Reference
	void setEvaluator(Evaluator evaluator){
		this.evaluator = evaluator;
	}
	
}
