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

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.PrintStream;
import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.Collection;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.NoSuchElementException;
import java.util.Queue;
import java.util.Set;
import java.util.SortedSet;
import java.util.TreeSet;
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

import com.google.gson.stream.JsonWriter;

import be.iminds.aiolos.info.NodeInfo;
import be.iminds.aiolos.platform.api.PlatformManager;
import be.iminds.iot.dianne.api.coordinator.AgentResult;
import be.iminds.iot.dianne.api.coordinator.Device;
import be.iminds.iot.dianne.api.coordinator.DianneCoordinator;
import be.iminds.iot.dianne.api.coordinator.EvaluationResult;
import be.iminds.iot.dianne.api.coordinator.Job;
import be.iminds.iot.dianne.api.coordinator.Job.Type;
import be.iminds.iot.dianne.api.coordinator.LearnResult;
import be.iminds.iot.dianne.api.coordinator.Notification;
import be.iminds.iot.dianne.api.coordinator.Notification.Level;
import be.iminds.iot.dianne.api.coordinator.Status;
import be.iminds.iot.dianne.api.dataset.DianneDatasets;
import be.iminds.iot.dianne.api.nn.eval.Evaluation;
import be.iminds.iot.dianne.api.nn.eval.Evaluator;
import be.iminds.iot.dianne.api.nn.learn.LearnProgress;
import be.iminds.iot.dianne.api.nn.learn.Learner;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkDTO;
import be.iminds.iot.dianne.api.nn.platform.DiannePlatform;
import be.iminds.iot.dianne.api.repository.DianneRepository;
import be.iminds.iot.dianne.api.rl.agent.Agent;
import be.iminds.iot.dianne.api.rl.agent.AgentProgress;
import be.iminds.iot.dianne.api.rl.learn.QLearnProgress;
import be.iminds.iot.dianne.coordinator.util.DianneCoordinatorWriter;

@SuppressWarnings("rawtypes")
@Component
public class DianneCoordinatorImpl implements DianneCoordinator {

	long boot = System.currentTimeMillis();
	
	BundleContext context;
	EventAdmin ea;

	DiannePlatform platform;
	DianneRepository repository;
	DianneDatasets datasets;
	
	PlatformManager aiolos;

	// separate queues for learn and eval jobs, 
	// makes sure eval jobs are not blocked by big learn jobs
	Queue<AbstractJob> queueLearn = new LinkedBlockingQueue<>();
	Queue<AbstractJob> queueEval = new LinkedBlockingQueue<>();
	Queue<AbstractJob> queueAct = new LinkedBlockingQueue<>();


	Set<AbstractJob> running = new HashSet<>();
	Queue<AbstractJob> finished = new CircularBlockingQueue<>(10);
	
	Map<UUID, Learner> learners = new ConcurrentHashMap<>();
	Map<UUID, Evaluator> evaluators = new ConcurrentHashMap<>();
	Map<UUID, Agent> agents = new ConcurrentHashMap<>();

	
	ExecutorService pool = Executors.newCachedThreadPool();

	Map<UUID, Device> devices = new ConcurrentHashMap<>();
	// keeps which device is doing what
	Map<UUID, BitSet> deviceUsage = new ConcurrentHashMap<>();
	// keeps failures to eventually blacklist device
	Map<UUID, Integer> deviceErrors = new ConcurrentHashMap<>();
	
	Queue<Notification> notifications = new CircularBlockingQueue<>(20);
	
	String storageDir = "jobs";
	
	@Override
	public Status getStatus(){
		int idle = deviceUsage.values().stream().mapToInt(t -> t.isEmpty() ? 1 : 0).sum();
		int learn = deviceUsage.values().stream().mapToInt(t -> t.get(Type.LEARN.ordinal()) ? 1 : 0).sum();
		int eval = deviceUsage.values().stream().mapToInt(t -> t.get(Type.EVALUATE.ordinal()) ? 1 : 0).sum();
		int act = deviceUsage.values().stream().mapToInt(t -> t.get(Type.ACT.ordinal()) ? 1 : 0).sum();

		
		long spaceLeft = repository.spaceLeft();
		Status currentStatus = new Status(queueAct.size()+queueLearn.size()+queueEval.size(), running.size(), learn, eval, act, idle, devices.size(), spaceLeft, boot);
		return currentStatus;
	}
	
	@Override
	public Promise<LearnResult> learn(String dataset, Map<String, String> config, NeuralNetworkDTO... nns) {
		for(NeuralNetworkDTO nn : nns)
			repository.storeNeuralNetwork(nn);
		
		LearnJob job = new LearnJob(this, dataset, config, nns);
		queueLearn.add(job);
		
		sendNotification(job.jobId, Level.INFO, "Learn job \""+job.name+"\" submitted.");
		
		schedule(Type.LEARN);
		
		return job.getPromise();
	}

	@Override
	public Promise<LearnResult> learn(String dataset, Map<String, String> config, String... nnName) {
		NeuralNetworkDTO[] nns = new NeuralNetworkDTO[nnName.length];
		for(int i=0;i<nns.length;i++){
			nns[i] = repository.loadNeuralNetwork(nnName[i]);
		}
		return learn(dataset, config, nns);
	}
	
	@Override
	public Promise<EvaluationResult> eval(String dataset, Map<String, String> config, NeuralNetworkDTO... nns) {
		if(nns != null){
			try {
				for(NeuralNetworkDTO nn : nns)
					repository.storeNeuralNetwork(nn);
			} catch(Exception e){
				// NN could be locked but still evaluation should be possible
			}
		}
		
		EvaluationJob job = new EvaluationJob(this, dataset, config, nns);
		queueEval.add(job);
		
		sendNotification(job.jobId, Level.INFO, "Evaluation job \""+job.name+"\" submitted.");
		
		schedule(Type.EVALUATE);
		
		return job.getPromise();
	}
	
	@Override
	public Promise<EvaluationResult> eval(String dataset, Map<String, String> config, String... nnName) {
		NeuralNetworkDTO[] nns = null;
		if(nnName != null){
			nns = new NeuralNetworkDTO[nnName.length];
			for(int i=0;i<nns.length;i++){
				nns[i] = repository.loadNeuralNetwork(nnName[i]);
			}
		}
		return eval(dataset, config, nns);
	}

	@Override
	public Promise<AgentResult> act(String dataset, Map<String, String> config, String... nnName) {
		NeuralNetworkDTO[] nns = null;
		if(nnName != null){
			nns = new NeuralNetworkDTO[nnName.length];
			for(int i=0;i<nns.length;i++){
				nns[i] = repository.loadNeuralNetwork(nnName[i]);
			}
		}

		return act(dataset, config, nns);
	}
	
	@Override
	public Promise<AgentResult> act(String dataset, Map<String, String> config, NeuralNetworkDTO... nns) {
		if(nns !=null){
			try {
				for(NeuralNetworkDTO nn : nns)
					repository.storeNeuralNetwork(nn);
			} catch(Exception e){
				// NN could be locked but still evaluation should be possible
			}
		}
		
		ActJob job = new ActJob(this, dataset, config, nns);
		queueAct.add(job);
		
		sendNotification(job.jobId, Level.INFO, "Act job \""+job.name+"\" submitted.");
		
		schedule(Type.ACT);
		
		return job.getPromise();
	}

	@Override
	public LearnResult getLearnResult(UUID jobId) {
		// check if this job is running, if so return progress
		AbstractJob running = getRunningJob(jobId);
		if(running!=null && running instanceof LearnJob){
			return ((LearnJob)running).getProgress();
		}
		
		// dig into the done results
		Object result = getResult(jobId);
		if(result instanceof LearnResult){
			return (LearnResult) result;
		}
		
		// noting found
		return null;
	}

	@Override
	public EvaluationResult getEvaluationResult(UUID jobId) {
		// check if this job is running, if so return progress
		AbstractJob running = getRunningJob(jobId);
		if(running!=null && running instanceof EvaluationJob){
			return ((EvaluationJob)running).getProgress();
		}
		
		// dig into the done results
		Object result = getResult(jobId);
		if(result instanceof EvaluationResult){
			return (EvaluationResult) result;
		}
		
		// nothing found
		return null;
	}

	@Override
	public AgentResult getAgentResult(UUID jobId) {
		// check if this job is running, if so return progress
		AbstractJob running = getRunningJob(jobId);
		if(running!=null && running instanceof ActJob){
			return ((ActJob)running).getProgress();
		}
		
		// dig into the done results
		Object result = getResult(jobId);
		if(result instanceof AgentResult){
			return (AgentResult) result;
		}
		
		// nothing found
		return null;
	}
	
	@Override
	public void stop(UUID jobId) throws Exception {
		AbstractJob job = getAbstractJob(jobId);
		if(job!=null){
			job.stop();
		}
	}
	
	private AbstractJob getAbstractJob(UUID jobId){
		AbstractJob job = null;
		try {
			job = queueLearn.stream().filter(j -> j.jobId.equals(jobId)).findFirst().get();
		} catch(NoSuchElementException e){}
		try {
			job = queueEval.stream().filter(j -> j.jobId.equals(jobId)).findFirst().get();
		} catch(NoSuchElementException e){}
		try {
			job = queueAct.stream().filter(j -> j.jobId.equals(jobId)).findFirst().get();
		} catch(NoSuchElementException e){}
		try {
			job = running.stream().filter(j -> j.jobId.equals(jobId)).findFirst().get();
		} catch(NoSuchElementException e){}
		try {
			job = finished.stream().filter(j -> j.jobId.equals(jobId)).findFirst().get();
		} catch(NoSuchElementException e){}
		
		return job;
	}
	
	@Override
	public Job getJob(UUID jobId){
		AbstractJob job = getAbstractJob(jobId);
		if(job!=null)
			return job.get();
		
		// TODO read from persistent storage if not in memory?
		
		return null;
	}
	
	@Override
	public List<Job> queuedJobs() {
		List<Job> learnJobs = queueLearn.stream().map(j -> j.get()).collect(Collectors.toList());
		List<Job> evalJobs = queueEval.stream().map(j -> j.get()).collect(Collectors.toList());
		List<Job> actJobs = queueAct.stream().map(j -> j.get()).collect(Collectors.toList());

		
		List<Job> allJobs = new ArrayList<>(learnJobs);
		allJobs.addAll(evalJobs);
		allJobs.addAll(actJobs);
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
		return new ArrayList<Device>(devices.values());
	}
	
	// called when a job is done
	void done(final AbstractJob<?> job) {
		// remove from running list
		if(running.remove(job)){
			job.targets.stream().forEach(uuid -> deviceUsage.get((UUID) uuid).set(job.type.ordinal(), false));
			
			try {
				// safe results to disc
				File dir = new File(storageDir+File.separator+job.jobId.toString());
				dir.mkdirs();
				
				File jobFile = new File(dir.getAbsolutePath()+File.separator+"job.json");
				
				try(JsonWriter writer = new JsonWriter(new FileWriter(jobFile))){
					writer.setLenient(true);
					writer.setIndent("  ");
					DianneCoordinatorWriter.writeJob(writer, job.get());
					writer.flush();
				} catch(Exception e){
					System.err.println("Failed to write job.json for Job "+job.jobId);
				}
				
				JobFailedException error = (JobFailedException) job.getPromise().getFailure();
				if(error !=null){
					File errorFile = new File(dir.getAbsolutePath()+File.separator+"error");
					try {
						error.printStackTrace(new PrintStream(new FileOutputStream(errorFile)));
					} catch (FileNotFoundException e) {
					}
					
					UUID device = error.getDevice();
					if(!deviceErrors.containsKey(device)){
						deviceErrors.put(device, 1);
					} else {
						int errors = deviceErrors.get(device);
						deviceErrors.put(device, errors+1);
					}
					
					File resultFile = new File(dir.getAbsolutePath()+File.separator+"progress.json");
					try(JsonWriter writer = new JsonWriter(new FileWriter(resultFile))){
						writer.setLenient(true);
						writer.setIndent("  ");
						DianneCoordinatorWriter.writeObject(writer, job.getProgress());
						writer.flush();
					} catch(Exception e){
						System.err.println("Failed to write job.json for Job "+job.jobId);
					}
					
					sendNotification(job.jobId, Level.DANGER, "Job \""+job.name+"\" failed on device "+device+" : "+error.getMessage() == null ? error.getClass().getName() : error.getMessage());
				} else {
					File resultFile = new File(dir.getAbsolutePath()+File.separator+"result.json");
					try(JsonWriter writer = new JsonWriter(new FileWriter(resultFile))){
						writer.setLenient(true);
						writer.setIndent("  ");
						DianneCoordinatorWriter.writeObject(writer, job.getPromise().getValue());
						writer.flush();
					} catch(Exception e){
						System.err.println("Failed to write job.json for Job "+job.jobId);
					}
					
					sendNotification(job.jobId, Level.SUCCESS, "Job \""+job.name+"\" finished successfully.");
				}
			} catch (InterruptedException e) {
			}
		} else {
			// if not running, remove from any queue
			queueLearn.remove(job);
			queueEval.remove(job);
			queueAct.remove(job);
			sendNotification(job.jobId, Level.WARNING, "Job \""+job.name+"\" canceled.");
		}
		
		finished.add(job);

		// schedule new one, check which queue has longest waiting queue item and schedule that one first
		SortedSet<Queue<AbstractJob>> queues = new TreeSet<>(new Comparator<Queue<AbstractJob>>() {
			@Override
			public int compare(Queue<AbstractJob> o1, Queue<AbstractJob> o2) {
				AbstractJob j1 = o1.peek();
				AbstractJob j2 = o2.peek();
				if(j1 == null){
					if(j2 ==null){
						return 0;
					} else {
						return 1;
					}
				} else if(j2 ==null){
					return -1;
				} else {
					return (int)(j1.submitted - j2.submitted);
				}
			}
		});
		queues.add(queueLearn);
		queues.add(queueEval);
		queues.add(queueAct);
		for(Queue q : queues){
			schedule( (q==queueLearn) ? Type.LEARN : (q==queueEval) ? Type.EVALUATE : Type.ACT  );
		}
	}
	
	// try to schedule the job on top of the queue
	synchronized void schedule(Type type){

		Queue<AbstractJob> queue = null;
		switch(type){
		case LEARN:
			queue = queueLearn;
			break;
		case EVALUATE:
			queue = queueEval;
			break;
		case ACT:
			queue = queueAct;
			break;
		}
		
		// try to schedule the next job on the queue
		AbstractJob<?> job = queue.peek();
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
		
		boolean forceFree = false;
		if(job.config.containsKey("forceFree")){
			forceFree = Boolean.parseBoolean("forceFree");
		}
		
		try {
			if(targets!=null){
				targets = findTargets(type, targets, filter, count, forceFree);
			} else if(type==Type.EVALUATE){
				targets = findTargets(type, evaluators.keySet(), filter, count, forceFree);
			} else if(type==Type.LEARN){
				targets = findTargets(type, learners.keySet(), filter, count, forceFree);
			} else if(type==Type.ACT){
				targets = findTargets(type, agents.keySet(), filter, count, forceFree);
			}
		} catch(Exception e){
			// impossible to find targets (i.e. not enough workers (yet)) ... try to schedule next and add this one to back of queue
			job = queue.poll();
			sendNotification(job.jobId, Level.WARNING, "Job \""+job.name+"\" failed to start: "+e.getMessage());
			schedule(type);
			queue.add(job);
			return;
		}
			
		for(UUID target : targets){
			deviceUsage.get(target).set(job.type.ordinal());
		}
		job = queue.poll();
		job.start(targets, pool);
		running.add(job);
			
		sendNotification(job.jobId, Level.INFO, "Job \""+job.name+"\" started.");

	}
	
	List<UUID> findTargets(Type type, Collection<UUID> ids, String filter, int count, boolean forceFree) throws Exception {
		List<UUID> candidates = ids.stream()
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
					// discourate the use of devices that tend to cause errors :-)
					.sorted(new Comparator<UUID>() {
						@Override
						public int compare(UUID o1, UUID o2) {
							int errors1 = deviceErrors.containsKey(o1) ? deviceErrors.get(o1) : 0;
							int errors2 = deviceErrors.containsKey(o2) ? deviceErrors.get(o2) : 0;
							return errors1 -errors2;
						}
					})
					.collect(Collectors.toList());
		
		// check if this is possible
		if(candidates.size() < count)
			throw new Exception("Insufficient infrastructure to meet the requirements of this Job");
		
		// check if the possible devices are currently available
		List<UUID> targets = candidates.stream()
			.filter(uuid -> deviceUsage.get(uuid).isEmpty()) // search for free nodes only
			.limit(count)
			.collect(Collectors.toList());
		
		// no free devices
		// search for devices not already running this job type and co-locate? 
		if(targets.size() != count && !forceFree){
			targets = candidates.stream()
					.filter(uuid -> !deviceUsage.get(uuid).get(type.ordinal())) 
					.limit(count)
					.collect(Collectors.toList());
		}

		if(targets.size() != count){
			throw new Exception("Not enough free targets for this Job");
		}
		
		return targets;
	}

	void sendNotification(UUID jobId, Level level, String message){
		Notification n = new Notification(jobId, level, message);
		
		Map<String, Object> properties = new HashMap<>();
		if(jobId!=null)
			properties.put("jobId", jobId);
		properties.put("level", n.level);
		properties.put("message", n.message);
		properties.put("timestamp", n.timestamp);
		
		String topic = (jobId != null) ? "dianne/jobs/"+jobId.toString() : "dianne/jobs";
		Event e = new Event(topic, properties);
		ea.postEvent(e);
		
		notifications.add(n);
	}

	void sendLearnProgress(UUID jobId, LearnProgress progress, Evaluation validation){
		Map<String, Object> properties = new HashMap<>();
		properties.put("jobId", jobId.toString());
		properties.put("iteration", progress.iteration);
		properties.put("minibatchLoss", progress.minibatchLoss);
		
		if(validation!=null)
			properties.put("validationLoss", validation.metric());
		
		if(progress instanceof QLearnProgress){
			properties.put("q", ((QLearnProgress)progress).q);
		}
		
		String topic = "dianne/jobs/"+jobId.toString()+"/progress";
		Event e = new Event(topic, properties);
		ea.postEvent(e);
	}
	
	void sendActProgress(UUID jobId, int worker, AgentProgress progress){
		Map<String, Object> properties = new HashMap<>();
		properties.put("jobId", jobId.toString());
		properties.put("sequence", progress.sequence);
		properties.put("reward", progress.reward);
		properties.put("worker", worker);
		
		String topic = "dianne/jobs/"+jobId.toString()+"/progress";
		Event e = new Event(topic, properties);
		ea.postEvent(e);
	}
	
	@Activate
	void activate(BundleContext context){
		this.context = context;
		
		String dir = context.getProperty("be.iminds.iot.dianne.job.storage");
		if(dir!=null){
			storageDir = dir;
		}

		File d = new File(storageDir);
		d.mkdirs();
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
	
	@Reference
	void setDianneDatasets(DianneDatasets datasets){
		this.datasets = datasets;
	}
	
	Device addDevice(UUID id){
		Device device = devices.get(id);
		if(device == null){
			deviceUsage.put(id, new BitSet(Type.values().length));

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
		return device;
	}
	
	void removeDevice(UUID id){
		if(id!=null){
			if(!learners.containsKey(id) 
			&& !evaluators.containsKey(id)
			&& !agents.containsKey(id)){
				deviceUsage.remove(id);
				devices.remove(id);
				deviceErrors.remove(id);
			}
		}
	}
	
	@Reference(policy=ReferencePolicy.DYNAMIC,
			cardinality=ReferenceCardinality.MULTIPLE)
	void addLearner(Learner learner, Map<String, Object> properties){
		UUID id = learner.getLearnerId();
		learners.put(id, learner);
		
		Device device = addDevice(id);
		device.learn = true;
		
		sendNotification(null, Level.INFO, "New Learner "+id+" is added to the system.");
		
		schedule(Type.LEARN);
	}
	
	void removeLearner(Learner learner, Map<String, Object> properties){
		UUID id = null;
		Iterator<Entry<UUID, Learner>> it = learners.entrySet().iterator();
		while(it.hasNext()){
			Entry<UUID, Learner> e = it.next();
			if(e.getValue() == learner){
				id = e.getKey();
				it.remove();
				break;
			}
		}
		
		removeDevice(id);
		
		sendNotification(null, Level.WARNING, "Learner "+id+" is removed from the system.");
		
		final UUID target = id;
		running.stream()
			.filter(job -> job.type == Type.LEARN)
			.filter(job -> job.targets.contains(target)).forEach(job -> job.done(new Exception("Job failed because executing node was killed")));
	}

	@Reference(policy=ReferencePolicy.DYNAMIC,
			cardinality=ReferenceCardinality.MULTIPLE)
	void addEvaluator(Evaluator evaluator, Map<String, Object> properties){
		UUID id = evaluator.getEvaluatorId();
		evaluators.put(id, evaluator);
		
		Device device = addDevice(id);
		device.eval = true;
		
		sendNotification(null, Level.INFO, "New Evaluator "+id+" is added to the system.");
		
		schedule(Type.EVALUATE);
	}
	
	void removeEvaluator(Evaluator evaluator, Map<String, Object> properties){
		UUID id = null;
		Iterator<Entry<UUID, Evaluator>> it = evaluators.entrySet().iterator();
		while(it.hasNext()){
			Entry<UUID, Evaluator> e = it.next();
			if(e.getValue() == evaluator){
				id = e.getKey();
				it.remove();
				break;
			}
		}
		
		removeDevice(id);
		
		sendNotification(null, Level.WARNING, "Evaluator "+id+" is removed from the system.");
		
		final UUID target = id;
		running.stream()
			.filter(job -> job.type == Type.EVALUATE)
			.filter(job -> job.targets.contains(target)).forEach(job -> job.done(new Exception("Job failed because executing node was killed")));
	}
	
	@Reference(policy=ReferencePolicy.DYNAMIC,
			cardinality=ReferenceCardinality.MULTIPLE)
	void addAgent(Agent agent, Map<String, Object> properties){
		UUID id = agent.getAgentId();
		this.agents.put(id, agent);
		
		Device device = addDevice(id);
		device.act = true;
		
		sendNotification(null, Level.INFO, "New Agent "+id+" is added to the system.");
		
		schedule(Type.ACT);
	}
	
	void removeAgent(Agent agent, Map<String, Object> properties){
		UUID id = null;
		Iterator<Entry<UUID, Agent>> it =this.agents.entrySet().iterator();
		while(it.hasNext()){
			Entry<UUID, Agent> e = it.next();
			if(e.getValue()==agent){
				id = e.getKey();
				it.remove();
				break;
			}
		}
		
		removeDevice(id);
		
		sendNotification(null, Level.WARNING, "Agent "+id+" is removed from the system.");
		
		final UUID target = id;
		running.stream()
			.filter(job -> job.type == Type.ACT)
			.filter(job -> job.targets.contains(target)).forEach(job -> job.done(new Exception("Job failed because executing node was killed")));
	}
	
	private AbstractJob getRunningJob(UUID jobId){
		try {
			return running.stream().filter(job -> job.jobId.equals(jobId)).findFirst().get();
		} catch(NoSuchElementException e){
			return null;
		}
	}
	
	private Object getResult(UUID jobId){
		// TODO read from persistent storage?
		try {
			AbstractJob j = finished.stream().filter(job -> job.jobId.equals(jobId)).findFirst().get();
			if(j.getPromise().getFailure() != null){
				// read progress until then
				return j.getProgress();
			} else {
				return j.getPromise().getValue();
			}
		} catch(Exception e){
			return null;
		}
	}
	
	boolean isRecurrent(NeuralNetworkDTO nn){
		if(nn.modules.values().stream().filter(module -> module.type.equals("Memory")).findAny().isPresent())
			return true;
		
		return nn.modules.values().stream().filter(module -> module.properties.get("category")!= null && module.properties.get("category").equals("Composite"))
			.mapToInt(module -> {	try {  
				return isRecurrent(repository.loadNeuralNetwork(module.properties.get("name"))) ? 1 : 0;
				//TODO if the composite is not inside the repository this might not work
			} catch(Exception e){ return 0;}}).sum() > 0;
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
