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

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Comparator;
import java.util.HashMap;
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
import org.osgi.framework.Filter;
import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;
import org.osgi.service.component.annotations.ReferenceCardinality;
import org.osgi.service.component.annotations.ReferencePolicy;
import org.osgi.service.event.Event;
import org.osgi.service.event.EventAdmin;
import org.osgi.util.promise.Promise;

import be.iminds.aiolos.info.NodeInfo;
import be.iminds.aiolos.monitor.node.api.NodeMonitorInfo;
import be.iminds.aiolos.platform.api.PlatformManager;
import be.iminds.iot.dianne.api.coordinator.Device;
import be.iminds.iot.dianne.api.coordinator.DianneCoordinator;
import be.iminds.iot.dianne.api.coordinator.EvaluationResult;
import be.iminds.iot.dianne.api.coordinator.Job;
import be.iminds.iot.dianne.api.coordinator.LearnResult;
import be.iminds.iot.dianne.api.coordinator.Notification;
import be.iminds.iot.dianne.api.coordinator.Notification.Level;
import be.iminds.iot.dianne.api.coordinator.Status;
import be.iminds.iot.dianne.api.nn.eval.Evaluator;
import be.iminds.iot.dianne.api.nn.learn.Learner;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkDTO;
import be.iminds.iot.dianne.api.nn.platform.DiannePlatform;
import be.iminds.iot.dianne.api.repository.DianneRepository;

@Component
public class DianneCoordinatorImpl implements DianneCoordinator {

	long boot = System.currentTimeMillis();
	
	BundleContext context;
	EventAdmin ea;

	DiannePlatform platform;
	DianneRepository repository;
	
	PlatformManager aiolos;

	// separate queues for learn and eval jobs, 
	// makes sure eval jobs are not blocked by big learn jobs
	Queue<AbstractJob> queueLearn = new LinkedBlockingQueue<>();
	Queue<AbstractJob> queueEval = new LinkedBlockingQueue<>();

	Set<AbstractJob> running = new HashSet<>();
	Queue<AbstractJob> finished = new LinkedBlockingQueue<>(10);
	
	Map<UUID, Learner> learners = new ConcurrentHashMap<>();
	Map<UUID, Evaluator> evaluators = new ConcurrentHashMap<>();
	
	ExecutorService pool = Executors.newCachedThreadPool();

	Map<UUID, Device> devices = new ConcurrentHashMap<>();
	// keeps which device is doing what
	// 0 = idle
	// 1 = learning
	// 2 = evaluating
	Map<UUID, Integer> deviceUsage = new ConcurrentHashMap<>(); 
	
	Queue<Notification> notifications = new LinkedBlockingQueue<>(20);
	
	@Override
	public Status getStatus(){
		int idle = deviceUsage.values().stream().mapToInt(i -> (i==0) ? 1 : 0).sum();
		int learn = deviceUsage.values().stream().mapToInt(i -> (i==1) ? 1 : 0).sum();
		int eval = deviceUsage.values().stream().mapToInt(i -> (i==2) ? 1 : 0).sum();

		long spaceLeft = repository.spaceLeft();
		Status currentStatus = new Status(queueLearn.size()+queueEval.size(), running.size(), learn, eval, idle, spaceLeft, boot);
		return currentStatus;
	}
	
	@Override
	public Promise<LearnResult> learn(NeuralNetworkDTO nn, String dataset, Map<String, String> config) {
		repository.storeNeuralNetwork(nn);
		
		LearnJob job = new LearnJob(this, nn, dataset, config);
		queueLearn.add(job);
		
		sendNotification(job.jobId, Level.INFO, "Job \""+job.name+"\" submitted.");
		
		schedule(false);
		
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
		queueEval.add(job);
		
		sendNotification(job.jobId, Level.INFO, "Job \""+job.name+"\" submitted.");
		
		schedule(true);
		
		return job.getPromise();
	}

	@Override
	public Promise<EvaluationResult> eval(String nnName, String dataset, Map<String, String> config) {
		return eval(repository.loadNeuralNetwork(nnName), dataset, config);
	}
	
	@Override
	public List<Job> queuedJobs() {
		List<Job> learnJobs = queueLearn.stream().map(j -> j.get()).collect(Collectors.toList());
		List<Job> evalJobs = queueEval.stream().map(j -> j.get()).collect(Collectors.toList());

		List<Job> allJobs = learnJobs;
		allJobs.addAll(evalJobs);
		allJobs.sort(new Comparator<Job>() {

			@Override
			public int compare(Job o1, Job o2) {
				return (int)(o1.submitted - o2.submitted);
			}
		});
		return allJobs;
	}

	@Override
	public List<Job> runningJobs() {
		return running.stream().map(j -> j.get()).collect(Collectors.toList());
	}

	@Override
	public List<Job> finishedJobs() {
		return finished.stream().map(j -> j.get()).collect(Collectors.toList());
	}
	
	
	@Override
	public List<Notification> getNotifications(){
		return new ArrayList<>(notifications);
	}

	@Override
	public List<Device> getDevices(){
		devices.values().stream().forEach(device -> {
			NodeMonitorInfo nmi = aiolos.getNodeMonitorInfo(device.id.toString());
			if(nmi!=null){
				device.cpuUsage = nmi.getCpuUsage();
				device.memUsage = nmi.getMemoryUsage();
			}
		});
		return new ArrayList<>(devices.values());
	}
	
	// called when a job is done
	void done(AbstractJob job){
		// remove from running list
		running.remove(job);
		job.targets.stream().forEach(uuid -> deviceUsage.put((UUID) uuid, 0));
		
		
		// TODO safe results to disc/archive?
		
		finished.add(job);
		
		sendNotification(job.jobId, Level.SUCCESS, "Job \""+job.name+"\" finished successfully.");

		// schedule new one, check which queue has longest waiting queue item and schedule that one first
		AbstractJob j1 = queueLearn.peek();
		if(j1==null){
			schedule(true);
		} else {
			AbstractJob j2 = queueEval.peek();
			if(j2 ==null){
				schedule(false);
			} else {
				if(j1.submitted < j2.submitted){
					schedule(true);
					schedule(false);
				} else {
					schedule(false);
					schedule(true);
				}
			}
		}
	}
	
	// try to schedule the job on top of the queue
	synchronized void schedule(boolean eval){
		// TODO what if not enough learners/evaluators or no matching learners/evaluators?
		
		Queue<AbstractJob> queue = eval ? queueEval : queueLearn;
		// try to schedule the next job on the queue
		AbstractJob job = queue.peek();
		if(job==null){
			// no more jobs...
			return;
		}
		
		// check in case a target list is given as comma separated uuids
		List<UUID> targets = null;
		String t = (String)job.config.get("targets");
		if(t!=null){
			try {
				targets = new ArrayList<>();
				for(String tt : t.split(",")){
					targets.add(UUID.fromString(tt));
				}
			} catch(Exception e){
				e.printStackTrace();
				targets = null;
			}
		}
		
		// check if count/filter is specified
		int count = 1;
		if(job.config.containsKey("targetCount")){
			count = Integer.parseInt((String)job.config.get("targetCount"));
		} else if(targets!=null){
			// if no count given but targets is, use all of them?
			count = targets.size();
		}
		String filter = (String)job.config.get("targetFilter");
		
		try {
			if(targets!=null){
				targets = findTargets(targets, filter, count);
			} else if(eval){
				targets = findTargets(evaluators.keySet(), filter, count);
			} else {
				targets = findTargets(learners.keySet(), filter, count);
			}
		} catch(Exception e){
			job = queue.poll();
			job.deferred.fail(e);
			
			sendNotification(job.jobId, Level.DANGER, "Job \""+job.name+"\" failed to start: "+e.getMessage());
		}
		
		if(targets==null){
			// what if no targets found? try next one or just keep on waiting?
			return;
		}
			
		int usage = eval ? 2 : 1;	
		for(UUID target : targets){
			deviceUsage.put(target, usage);
		}
		job = queue.poll();
		job.start(targets, pool);
		running.add(job);
			
		sendNotification(job.jobId, Level.INFO, "Job \""+job.name+"\" started.");

	}
	
	List<UUID> findTargets(Collection<UUID> ids, String filter, int count) throws Exception {
		List<UUID> targets = ids.stream()
					.map(uuid -> devices.get(uuid))
					.filter( device -> {	// match device filter
						if(device==null)
							return false;  // could be an invalid uuid if given from targets property
						
						if(filter==null)
							return true;
						
						try {
							Filter f = context.createFilter(filter);
							return f.matches(toMap(device));
						} catch(Exception e){
							e.printStackTrace();
							return false;
						}
					})
					.map(device -> device.id)
					.collect(Collectors.toList());
		// check if this is possible
		if(targets.size() < count)
			throw new Exception("Insufficient infrastructure to meet the requirements of this Job");
		
		// check if the possible devices are currently available
		targets = targets.stream()
			.filter(uuid -> deviceUsage.get(uuid)==0) // only free nodes can be selected
			.limit(count)
			.collect(Collectors.toList());

		if(targets.size() != count){
			return null;
		}
		
		return targets;
	}

	void sendNotification(UUID jobId, Level level, String message){
		Notification n = new Notification(level, message);
		
		Map<String, Object> properties = new HashMap<>();
		properties.put("level", n.level);
		properties.put("message", n.message);
		properties.put("timestamp", n.timestamp);
		
		String topic = (jobId != null) ? "dianne/jobs/"+jobId.toString() : "dianne/jobs";
		Event e = new Event(topic, properties);
		ea.postEvent(e);
		
		notifications.add(n);
	}

	@Activate
	void activate(BundleContext context){
		this.context = context;
	}
	
	// TODO here we use a name (AIOLOS) that is alphabetically before the others
	// so that the reference is set in addLearer/addEvaluator
	@Reference(cardinality=ReferenceCardinality.OPTIONAL)
	void setAIOLOS(PlatformManager p){  
		this.aiolos = p;
	}
	
	@Reference
	void setEA(EventAdmin ea){
		this.ea = ea;
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
		
		Device device = devices.get(id);
		if(device == null){
			deviceUsage.put(id, 0);

			String name = id.toString();
			String arch = "unknown";
			String os = "unknown";
			String ip = "unknown";
			
			if(aiolos!=null){
				NodeInfo n = aiolos.getNode(id.toString());
				name = n.getName();
				arch = n.getArch();
				os = n.getOS();
				ip = n.getIP();
			}
			device = new Device(id, name, arch, os, ip);
			devices.put(id, device);
		}
		device.learn = true;
		
		sendNotification(null, Level.INFO, "New Learner "+id+" is added to the system.");
		
		schedule(false);
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
				deviceUsage.remove(id);
				devices.remove(id);
			}
		}
		
		sendNotification(null, Level.WARNING, "Learner "+id+" is removed from the system.");

	}

	@Reference(policy=ReferencePolicy.DYNAMIC,
			cardinality=ReferenceCardinality.MULTIPLE)
	void addEvaluator(Evaluator evaluator, Map<String, Object> properties){
		UUID id = evaluator.getEvaluatorId();
		this.evaluators.put(id, evaluator);
		
		Device device = devices.get(id);
		if(device == null){
			deviceUsage.put(id, 0);

			String name = id.toString();
			String arch = "unknown";
			String os = "unknown";
			String ip = "unknown";
			
			if(aiolos!=null){
				NodeInfo n = aiolos.getNode(id.toString());
				name = n.getName();
				arch = n.getArch();
				os = n.getOS();
				ip = n.getIP();
			}
			device = new Device(id, name, arch, os, ip);
			devices.put(id, device);
		}
		device.eval = true;
		
		sendNotification(null, Level.INFO, "New Evaluator "+id+" is added to the system.");
		
		schedule(true);
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
				deviceUsage.remove(id);
				devices.remove(id);
			}
		}
		
		sendNotification(null, Level.WARNING, "Evaluator "+id+" is removed from the system.");

	}
	
	boolean isRecurrent(NeuralNetworkDTO nn){
		if(nn.modules.values().stream().filter(module -> module.type.equals("Memory")).findAny().isPresent())
			return true;
		
		return nn.modules.values().stream().filter(module -> module.properties.get("category").equals("Composite"))
			.mapToInt(module ->  
				isRecurrent(repository.loadNeuralNetwork(module.properties.get("name"))) ? 1 : 0).sum() > 0;
	}
	
	// TODO use Object Conversion spec for this...
	private Map<String, Object> toMap(Object o){
		Map<String, Object> properties = new HashMap<>();
		for(Field f : o.getClass().getFields()){
			try {
				properties.put(f.getName(), f.get(o));
			} catch (Exception e) {
			}
		}
		return properties;
	}
}
