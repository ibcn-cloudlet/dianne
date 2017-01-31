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
	
	protected final String[] nnNames;
	protected final NeuralNetworkDTO[] nns;
	protected final String dataset;
	protected final Map<String, String> config;

	protected final Deferred<T> deferred = new Deferred<>();

	protected List<UUID> targets = new ArrayList<>();
	protected Map<UUID, NeuralNetworkInstanceDTO[]> nnis = new HashMap<>();;
	protected Map<UUID, UUID> targetsByNNi = new HashMap<>();
	
	protected long submitted = 0;
	protected long started = 0;
	protected long stopped = 0;
	
	public AbstractJob(DianneCoordinatorImpl coord,
			Type type,
			String d,
			Map<String, String> c,
			NeuralNetworkDTO[] nns){
		this.jobId = UUID.randomUUID();
		
		if(c.containsKey("name")){
			this.name = c.get("name");
		} else {
			this.name = jobId.toString();
		}
	
		this.type = type;
		
		this.coordinator = coord;
		
		this.nns = nns;
		if(nns!=null){
			this.nnNames = new String[nns.length];
			for(int i=0;i<nns.length;i++){
				nnNames[i] = nns[i].name;
			}
		} else {
			this.nnNames = null;
		}
		this.dataset = d;
		this.config = c;
		
		this.submitted = System.currentTimeMillis();
	}
	
	public void run(){
		started = System.currentTimeMillis();
		try {
			// deploy each neural network on each target instance
			// TODO do we indeed need every network on each target?
			// how to specify custom deployments?
			if(nns != null){
				nnis = new HashMap<>();
				targetsByNNi = new HashMap<>();
				for(UUID target : targets){
					NeuralNetworkInstanceDTO[] instances = new NeuralNetworkInstanceDTO[nns.length];
					for(int i=0;i<nns.length;i++){
						try {
							NeuralNetworkDTO nn = nns[i];
							NeuralNetworkInstanceDTO nni = coordinator.platform.deployNeuralNetwork(nn.name, "Dianne Coordinator "+type+" job "+jobId, this.config, target);
							instances[i] = nni;
							targetsByNNi.put(nni.id, target);
						} catch(Throwable t){
							throw new JobFailedException(target, AbstractJob.this.jobId, "Failed to deploy NN instances: "+t.getMessage(), t);
						}
					}
					nnis.put(target, instances);
				}
			}
			
			// execute
			execute();
		} catch(JobFailedException e){
			done(e);
		}
	}
	
	// to be implemented by the actual Job
	public abstract void execute() throws JobFailedException;
	
	public abstract T getProgress();
	
	public void cleanup() {};
	
	public void stop() throws Exception {
		if(started > 0){
			throw new Exception("This job cannot be stopped");
		} else {
			done(new JobFailedException(null, this.jobId, "Job "+this.jobId+" cancelled.", null));
		}
	}
	
	// to be called when the execution is done
	public void done(T result){
		deferred.resolve(result);
		done();
	}
	
	// to be called on error
	public void done(JobFailedException error){
		deferred.fail(error);
		done();
	}
	
	private void done(){
		stopped = System.currentTimeMillis();
		
		try {
			cleanup();
			
			// undeploy neural networks on target instances
			for(NeuralNetworkInstanceDTO[] instances : nnis.values()){
				for(NeuralNetworkInstanceDTO nni : instances){
					coordinator.platform.undeployNeuralNetwork(nni);
				}
			}
		
		} finally {
			// this one is free again for the coordinator
			coordinator.done(this);
		}
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
		Job job = new Job(jobId, name, type, category, dataset, config, nnNames);
		job.submitted = submitted;
		job.started = started;
		job.stopped = stopped;
		job.targets = targets;
		return job;
	}

}
