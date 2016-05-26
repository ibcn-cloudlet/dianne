package be.iminds.iot.dianne.coordinator;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.Executor;

import org.osgi.util.promise.Deferred;
import org.osgi.util.promise.Promise;

import be.iminds.iot.dianne.api.coordinator.Job;
import be.iminds.iot.dianne.api.coordinator.Job.Category;
import be.iminds.iot.dianne.api.coordinator.Job.Type;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkDTO;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkInstanceDTO;

public abstract class AbstractJob<T> implements Runnable {

	protected final UUID jobId;
	protected final String name;
	protected final Type type;
	protected Category category;
	
	protected final DianneCoordinatorImpl coordinator;
	
	protected final NeuralNetworkDTO nn;
	protected final String dataset;
	protected final Map<String, String> config;

	protected final Deferred<T> deferred = new Deferred<>();

	protected List<UUID> targets = new ArrayList<>();
	protected Map<UUID, NeuralNetworkInstanceDTO> nnis = new HashMap<>();;
	protected Map<UUID, UUID> targetsByNNi = new HashMap<>();
	
	protected long submitted = 0;
	protected long started = 0;
	protected long stopped = 0;
	
	public AbstractJob(DianneCoordinatorImpl coord,
			Type type,
			NeuralNetworkDTO nn,
			String d,
			Map<String, String> c){
		this.jobId = UUID.randomUUID();
		
		if(c.containsKey("name")){
			this.name = c.get("name");
		} else {
			this.name = jobId.toString();
		}
	
		this.type = type;
		
		this.coordinator = coord;
		
		this.nn = nn;
		this.dataset = d;
		this.config = c;
		
		this.submitted = System.currentTimeMillis();
	}
	
	public void run(){
		started = System.currentTimeMillis();
		try {
			// deploy neural network on each target instance
			nnis = new HashMap<>();
			targetsByNNi = new HashMap<>();
			for(UUID target : targets){
				NeuralNetworkInstanceDTO nni = coordinator.platform.deployNeuralNetwork(nn.name, "Dianne Coordinator LearnJob "+jobId, target);
				nnis.put(target, nni);
				targetsByNNi.put(nni.id, target);
			}
			
			// execute
			execute();
		} catch(Throwable t){
			done(t);
		}
	}
	
	// to be implemented by the actual Job
	public abstract void execute() throws Exception;
	
	public abstract T getProgress();
	
	public void cleanup() {};
	
	public void stop() throws Exception {
		if(started > 0){
			throw new Exception("This job cannot be stopped");
		} else {
			done(new Exception("Job "+this.jobId+" cancelled."));
		}
	}
	
	// to be called when the execution is done
	public void done(T result){
		deferred.resolve(result);
		done();
	}
	
	// to be called on error
	public void done(Throwable error){
		deferred.fail(error);
		done();
	}
	
	private void done(){
		stopped = System.currentTimeMillis();
		
		cleanup();
		
		// undeploy neural networks on target instances here?
		for(NeuralNetworkInstanceDTO nni : nnis.values()){
			coordinator.platform.undeployNeuralNetwork(nni);
		}
		
		// this one is free again for the coordinator
		coordinator.done(this);
	}
	
	public void start(List<UUID> targets){
		this.targets = targets;
		Thread t = new Thread(this);
		t.start();
	}
	
	public void start(List<UUID> targets, Executor pool){
		this.targets = targets;
		pool.execute(this);
	}
	
	public Promise<T> getPromise(){
		return deferred.getPromise();
	}
	
	public boolean isRunning(){
		return started > 0 && stopped == 0;
	}
	
	public boolean isDone(){
		return stopped > 0;
	}
	
	public Job get(){
		Job job = new Job(jobId, name, type, category, nn.name, dataset, config);
		job.submitted = submitted;
		job.started = started;
		job.stopped = stopped;
		job.targets = targets;
		return job;
	}

}
