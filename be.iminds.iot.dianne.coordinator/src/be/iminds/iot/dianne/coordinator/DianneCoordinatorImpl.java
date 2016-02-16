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
import java.util.List;
import java.util.Map;
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
		
		// TODO safe results to disc/archive?
		
		// schedule new one
		schedule();
	}
	
	void schedule(){
		// TODO what if not enough learners/evaluators or no matching learners/evaluators?
		
		// try to schedule the next job on the queue
		AbstractJob job = queue.peek();
		// TODO check the config whether this one can execute
		if(job instanceof LearnJob){
			// search free learner
			// TODO keep the free/occupied learners
			Learner l = learners.values().stream().filter(learner -> learner.isBusy() == false).findFirst().get();
			ArrayList<UUID> targets = new ArrayList<>();
			targets.add(l.getLearnerId());
			queue.poll().start(targets, pool);
		} if(job instanceof EvaluationJob){
			// search free evaluator
			// TODO keep the free/occupied learners
			// TODO isBusy for evaluators?
			Evaluator e = evaluators.values().stream().findFirst().get();
			ArrayList<UUID> targets = new ArrayList<>();
			targets.add(e.getEvaluatorId());
			queue.poll().start(targets, pool);
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
		this.learners.put(learner.getLearnerId(), learner);
	}
	
	void removeLearner(Learner learner, Map<String, Object> properties){
		this.learners.entrySet().remove(learner);
	}

	@Reference(policy=ReferencePolicy.DYNAMIC,
			cardinality=ReferenceCardinality.MULTIPLE)
	void addEvaluator(Evaluator evaluator, Map<String, Object> properties){
		this.evaluators.put(evaluator.getEvaluatorId(), evaluator);
	}
	
	void removeEvaluator(Evaluator evaluator, Map<String, Object> properties){
		this.evaluators.entrySet().remove(evaluator);
	}
	
	
}
