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
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Queue;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.stream.Collectors;

import org.osgi.framework.BundleContext;
import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;
import org.osgi.service.component.annotations.ReferenceCardinality;
import org.osgi.service.component.annotations.ReferencePolicy;
import org.osgi.util.promise.Promise;

import be.iminds.iot.dianne.api.coordinator.DianneCoordinator;
import be.iminds.iot.dianne.api.coordinator.EvaluationResult;
import be.iminds.iot.dianne.api.coordinator.Job;
import be.iminds.iot.dianne.api.coordinator.LearnResult;
import be.iminds.iot.dianne.api.nn.eval.Evaluator;
import be.iminds.iot.dianne.api.nn.learn.Learner;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkDTO;
import be.iminds.iot.dianne.api.nn.platform.DiannePlatform;
import be.iminds.iot.dianne.api.repository.DianneRepository;

@Component
public class DianneCoordinatorImpl implements DianneCoordinator {

	BundleContext context;

	DiannePlatform platform;
	DianneRepository repository;

	Queue<AbstractJob> queue = new LinkedBlockingQueue<>();
	Set<AbstractJob> running = new HashSet<>();
	
	Map<UUID, Learner> learners = new ConcurrentHashMap<>();
	Map<UUID, Evaluator> evaluators = new ConcurrentHashMap<>();
	
	ExecutorService pool = Executors.newCachedThreadPool();
	
	Map<UUID, Boolean> machines = new ConcurrentHashMap<>(); 
	
	@Override
	public Promise<LearnResult> learn(NeuralNetworkDTO nn, String dataset, Map<String, String> config) {
		repository.storeNeuralNetwork(nn);
		
		LearnJob job = new LearnJob(this, nn, dataset, config);
		queue.add(job);
		schedule();
		
		return job.getPromise();
	}

	@Override
	public Promise<LearnResult> learn(String nnName, String dataset, Map<String, String> config) {
		return learn(repository.loadNeuralNetwork(nnName), dataset, config);
	}
	
	@Override
	public Promise<EvaluationResult> eval(NeuralNetworkDTO nn, String dataset, Map<String, String> config) {
		// TODO evaluate time on other/multiple platforms?
		repository.storeNeuralNetwork(nn);
		
		EvaluationJob job = new EvaluationJob(this, nn, dataset, config);
		queue.add(job);
		schedule();
		
		return job.getPromise();
	}

	@Override
	public Promise<EvaluationResult> eval(String nnName, String dataset, Map<String, String> config) {
		return eval(repository.loadNeuralNetwork(nnName), dataset, config);
	}
	
	@Override
	public List<Job> queuedJobs() {
		return queue.stream().map(j -> j.get()).collect(Collectors.toList());
	}

	@Override
	public List<Job> runningJobs() {
		return running.stream().map(j -> j.get()).collect(Collectors.toList());
	}

	@Override
	public List<Job> allJobs() {
		List<Job> all = new ArrayList<>();
		all.addAll(queuedJobs());
		all.addAll(runningJobs());
		return all;
	}
	
	
	// try to do next job
	void done(AbstractJob job){
		// remove from running list
		running.remove(job);
		job.targets.stream().forEach(uuid -> machines.put((UUID) uuid, false));
		
		// TODO safe results to disc/archive?
		
		// schedule new one
		schedule();
	}
	
	synchronized void schedule(){
		// TODO what if not enough learners/evaluators or no matching learners/evaluators?
		
		// try to schedule the next job on the queue
		AbstractJob job = queue.peek();
		// TODO check the config whether this one can execute
		if(job instanceof LearnJob){
			// search free learner
			// TODO collect multiple in case multiple learners required
			int required = 1;
			// TODO filter learners on job properties
			List<UUID> targets = learners.keySet().stream().filter(uuid -> !machines.get(uuid)).limit(required).collect(Collectors.toList());
			if(targets.size()!=required)
				return;
			
			for(UUID target : targets){
				machines.put(target, true);
			}
			job = queue.poll();
			job.start(targets, pool);
			running.add(job);
		} if(job instanceof EvaluationJob){
			// search free evaluator
			// TODO collect multiple in case multiple evaluators required
			int required = 1;
			// TODO filter evaluators on job properties
			List<UUID> targets = evaluators.keySet().stream().filter(uuid -> !machines.get(uuid)).limit(required).collect(Collectors.toList());
			if(targets.size()!=required)
				return;

			for(UUID target : targets){
				machines.put(target, true);
			}
			job = queue.poll();
			job.start(targets, pool);
			running.add(job);
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
	
	@Reference(policy=ReferencePolicy.DYNAMIC,
			cardinality=ReferenceCardinality.MULTIPLE)
	void addLearner(Learner learner, Map<String, Object> properties){
		UUID id = learner.getLearnerId();
		this.learners.put(id, learner);
		
		if(!machines.containsKey(id)){
			machines.put(id, false);
		}
		
		schedule();
	}
	
	void removeLearner(Learner learner, Map<String, Object> properties){
		UUID id = null;
		Iterator<Entry<UUID, Learner>> it =this.learners.entrySet().iterator();
		while(it.hasNext()){
			Entry<UUID, Learner> e = it.next();
			if(e.getValue()==learner){
				id = e.getKey();
				it.remove();
				break;
			}
		}
		
		if(id!=null){
			if(!learners.containsKey(id) 
				&& !evaluators.containsKey(id)){
				machines.remove(id);
			}
		}
	}

	@Reference(policy=ReferencePolicy.DYNAMIC,
			cardinality=ReferenceCardinality.MULTIPLE)
	void addEvaluator(Evaluator evaluator, Map<String, Object> properties){
		UUID id = evaluator.getEvaluatorId();
		this.evaluators.put(id, evaluator);
		
		if(!machines.containsKey(id)){
			machines.put(id, false);
		}
		
		schedule();
	}
	
	void removeEvaluator(Evaluator evaluator, Map<String, Object> properties){
		UUID id = null;
		Iterator<Entry<UUID, Evaluator>> it =this.evaluators.entrySet().iterator();
		while(it.hasNext()){
			Entry<UUID, Evaluator> e = it.next();
			if(e.getValue()==evaluator){
				id = e.getKey();
				it.remove();
				break;
			}
		}
		
		if(id!=null){
			if(!learners.containsKey(id) 
				&& !evaluators.containsKey(id)){
				machines.remove(id);
			}
		}
	}
	
	
	
	boolean isRecurrent(NeuralNetworkDTO nn){
		if(nn.modules.values().stream().filter(module -> module.type.equals("Memory")).findAny().isPresent())
			return true;
		
		return nn.modules.values().stream().filter(module -> module.properties.get("category").equals("Composite"))
			.mapToInt(module ->  
				isRecurrent(repository.loadNeuralNetwork(module.properties.get("name"))) ? 1 : 0).sum() > 0;
	}
}
