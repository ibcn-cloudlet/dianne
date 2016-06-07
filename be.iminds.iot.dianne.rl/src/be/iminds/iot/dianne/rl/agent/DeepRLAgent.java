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
package be.iminds.iot.dianne.rl.agent;

import java.util.ArrayList;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;

import org.osgi.framework.BundleContext;
import org.osgi.framework.Constants;
import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Deactivate;
import org.osgi.service.component.annotations.Reference;
import org.osgi.service.component.annotations.ReferenceCardinality;
import org.osgi.service.component.annotations.ReferencePolicy;

import be.iminds.iot.dianne.api.nn.Dianne;
import be.iminds.iot.dianne.api.nn.NeuralNetwork;
import be.iminds.iot.dianne.api.nn.module.Module.Mode;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkInstanceDTO;
import be.iminds.iot.dianne.api.rl.agent.Agent;
import be.iminds.iot.dianne.api.rl.agent.AgentProgress;
import be.iminds.iot.dianne.api.rl.dataset.ExperiencePool;
import be.iminds.iot.dianne.api.rl.dataset.ExperiencePoolSample;
import be.iminds.iot.dianne.api.rl.environment.Environment;
import be.iminds.iot.dianne.nn.util.DianneConfigHandler;
import be.iminds.iot.dianne.rl.agent.config.AgentConfig;
import be.iminds.iot.dianne.rl.agent.strategy.ActionStrategy;
import be.iminds.iot.dianne.tensor.Tensor;

@Component
public class DeepRLAgent implements Agent {

	private UUID agentId;
	
	private Map<String, ExperiencePool> pools = new HashMap<String, ExperiencePool>();
	private Map<String, Environment> envs = new HashMap<String, Environment>();
	private Map<String, ActionStrategy> strategies = new HashMap<String, ActionStrategy>();
	private Dianne dianne;

	private NeuralNetwork nn;
	private ExperiencePool pool;
	private Environment env;
	
	private AgentConfig config;
	private Map<String, String> properties;
	
	private Thread actingThread;
	private long i = 0;
	private volatile boolean acting;
	
	private ActionStrategy actionStrategy;
	
	// separate thread for updating the experience pool
	private Thread experienceUploadThread;
	private List<ExperiencePoolSample> buffer;
	private List<ExperiencePoolSample> upload;
	private volatile boolean bufferReady = false;
	private volatile boolean uploading = false;
	
	@Reference
	void setDianne(Dianne d){
		dianne = d;
	}
	
	@Reference(cardinality = ReferenceCardinality.MULTIPLE, policy = ReferencePolicy.DYNAMIC)
	void addExperiencePool(ExperiencePool pool, Map<String, Object> properties) {
		String name = (String) properties.get("name");
		this.pools.put(name, pool);
	}

	void removeExperiencePool(ExperiencePool pool, Map<String, Object> properties) {
		String name = (String) properties.get("name");
		this.pools.remove(name);
	}

	@Reference(cardinality = ReferenceCardinality.MULTIPLE, policy = ReferencePolicy.DYNAMIC)
	void addEnvironment(Environment env, Map<String, Object> properties) {
		String name = (String) properties.get("name");
		this.envs.put(name, env);
	}

	void removeEnvironment(Environment env, Map<String, Object> properties) {
		String name = (String) properties.get("name");
		this.envs.remove(name);
	}

	@Reference(cardinality = ReferenceCardinality.MULTIPLE, policy = ReferencePolicy.DYNAMIC)
	void addStrategy(ActionStrategy s, Map<String, Object> properties) {
		String strategy = (String) properties.get("strategy");
		this.strategies.put(strategy, s);
	}

	void removeStrategy(ActionStrategy s, Map<String, Object> properties) {
		String strategy = (String) properties.get("strategy");
		this.strategies.remove(strategy);
	}
	
	@Activate
	public void activate(BundleContext context){
		this.agentId = UUID.fromString(context.getProperty(Constants.FRAMEWORK_UUID));
	}
	
	@Deactivate
	void deactivate() {
		if(acting)
			stop();
	}

	@Override
	public UUID getAgentId(){
		return agentId;
	}
	
	@Override
	public boolean isBusy(){
		return acting;
	}
	
	@Override
	public synchronized void act(String experiencePool, Map<String, String> config, NeuralNetworkInstanceDTO nni, String environment)
			throws Exception {
		if (acting)
			throw new Exception("Already running an Agent here");
		else if (environment == null || !envs.containsKey(environment))
			throw new Exception("Environment " + environment + " is null or not available");
		else if (experiencePool != null && !pools.containsKey(experiencePool))
			throw new Exception("ExperiencePool " + experiencePool + " is not available");

		
		System.out.println("Agent Configuration");
		System.out.println("===================");

		this.config = DianneConfigHandler.getConfig(config, AgentConfig.class);
		this.properties = config;
		
		String strategy = this.config.strategy.toString();
		actionStrategy = strategies.get(strategy);
		if(actionStrategy==null)
			throw new RuntimeException("Invalid strategy selected: "+strategy);
		
		actionStrategy.configure(config);
		
		try {
			nn = dianne.getNeuralNetwork(nni).getValue();
		} catch(Exception e){}
		if (nn == null)
			throw new Exception("Network instance " + nni.id + " is not available");
		
		nn.getInput().setMode(EnumSet.of(Mode.BLOCKING));
		
		env = envs.get(environment);
		pool = pools.get(experiencePool);

		buffer = new ArrayList<>(this.config.experienceInterval);
		upload = new ArrayList<>(this.config.experienceInterval);
		for(int i=0;i<this.config.experienceInterval;i++){
			buffer.add(new ExperiencePoolSample());
			upload.add(new ExperiencePoolSample());
		}
		
		actingThread = new Thread(new AgentRunnable());
		experienceUploadThread = new Thread(new UploadRunnable());
		acting = true;
		actingThread.start();
		experienceUploadThread.start();
	}

	@Override
	public AgentProgress getProgress() {
		return new AgentProgress(i);
	}
	
	@Override
	public synchronized void stop() {
		try {
			if (actingThread != null && actingThread.isAlive()) {
				acting = false;
				actingThread.join();
			}
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
	}
	
	private Tensor selectActionFromObservation(Tensor state, long i) {
		Tensor out = nn.forward(state);
		return actionStrategy.selectActionFromOutput(out, i);
	}
	
	private class AgentRunnable implements Runnable {

		@Override
		public void run() {
			Tensor current = new Tensor();
			Tensor next = new Tensor();
			ExperiencePoolSample s = new ExperiencePoolSample();
			
			env.setup(properties);
			
			try {
				s.input = env.getObservation(current);
	
				for(i = 0; acting; i++) {
					if(config.syncInterval > 0 && i % config.syncInterval == 0){
						// sync parameters
						try {
							nn.loadParameters(config.tag);
						} catch(Exception e){
							System.out.println("Failed to load parameters with tag "+config.tag);
						}
					}
					
					s.output = selectActionFromObservation(s.input, i);
	
					s.reward= env.performAction(s.output);
					s.nextState = env.getObservation(next);
					if(s.nextState == null){
						s.isTerminal = true;
					} else {
						s.isTerminal = false;
					}
	
					// upload in batch
					ExperiencePoolSample b = buffer.get((int)(i % config.experienceInterval));
					b.input = s.input.copyInto(b.input);
					b.output = s.output.copyInto(b.output);
					b.reward = s.reward;
					b.isTerminal = s.isTerminal;
					if(!s.isTerminal){
						b.nextState = s.nextState.copyInto(b.nextState);
					}
					
					if(i > 0 && (i+1) % config.experienceInterval == 0){
						// buffer full, switch to upload
						// check if upload finished
						// if still uploading ... wait now
						if(uploading){
							synchronized(upload){
								if(uploading){
									try {
										upload.wait();
									} catch (InterruptedException e) {
									}
								}
							}
						}
						List<ExperiencePoolSample> temp = upload;
						upload = buffer;
						buffer = temp;
						bufferReady = true;
						if(!uploading){
							synchronized(upload){
								upload.notifyAll();
							}
						}
					}
					
	
					// if nextObservation was null, this is a terminal state - reset environment and start over
					if(s.nextState == null){
						env.reset();
						s.input = env.getObservation(current);
					} else {
						s.input = next.copyInto(current);
					}
				}
			} catch(Exception e){
				e.printStackTrace();
			}
			
			env.cleanup();
			
			acting = false;
		}
	}
	
	private class UploadRunnable implements Runnable {

		@Override
		public void run() {
			if(config.clean){
				if(pool!=null)
					pool.reset();
			}
			
			while(acting){
				// wait till new buffer is ready
				if(!bufferReady){
					synchronized(buffer){
						if(!bufferReady){
							try {
								buffer.wait();
							} catch (InterruptedException e) {
								e.printStackTrace();
							}
						}
					}
				}

				bufferReady = false;
				uploading = true;
				if(pool!=null){
					pool.addSamples(upload);
				}
				
				uploading = false;
				
				synchronized(upload){
					upload.notifyAll();
				}
				
			}
		}
		
	}
}
