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
import be.iminds.iot.dianne.api.rl.dataset.ExperiencePool;
import be.iminds.iot.dianne.api.rl.dataset.ExperiencePoolSample;
import be.iminds.iot.dianne.api.rl.environment.Environment;
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
	
	private Thread actingThread;
	private volatile boolean acting;
	private int syncInterval = 10000;
	private int gcInterval = 1000;
	
	// separate thread for updating the experience pool
	private Thread experienceUploadThread;
	private int experienceInterval = 1000; // update in batches
	private int experienceSize = 1000000; // maximum size in experience pool
	private List<ExperiencePoolSample> samples = new ArrayList<ExperiencePoolSample>();

	private String tag = "run";
	private boolean clean = false;
	
	private ActionStrategy actionStrategy;
	
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

		if(config.containsKey("tag"))
			tag = config.get("tag"); 
		
		if(config.containsKey("gcInterval")){
			gcInterval = Integer.parseInt(config.get("gcInterval"));
		}
		
		if (config.containsKey("clean"))
			clean = Boolean.parseBoolean(config.get("clean"));
		
		if (config.containsKey("syncInterval"))
			syncInterval = Integer.parseInt(config.get("syncInterval"));
		
		if (config.containsKey("experienceInterval"))
			experienceInterval = Integer.parseInt(config.get("experienceInterval"));
		
		if (config.containsKey("experienceSize"))
			experienceSize = Integer.parseInt(config.get("experienceSize"));
		
		String strategy = "greedy";
		if(config.containsKey("strategy"))
			strategy = config.get("strategy");
		
		System.out.println("Agent Configuration");
		System.out.println("===================");
		System.out.println("* tag = "+tag);
		System.out.println("* strategy = "+strategy);
		System.out.println("* clean = "+clean);
		System.out.println("* syncInterval = "+syncInterval);
		System.out.println("* gcInterval = "+gcInterval);
		System.out.println("* experienceInterval = "+experienceInterval);
		System.out.println("* experienceSize = "+experienceSize);
		System.out.println("---");
		
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

		actingThread = new Thread(new AgentRunnable());
		experienceUploadThread = new Thread(new UploadRunnable());
		acting = true;
		actingThread.start();
		experienceUploadThread.start();
	}

	@Override
	public synchronized void stop() {
		try {
			if (actingThread != null && actingThread.isAlive()) {
				acting = false;
				actingThread.join();
				experienceUploadThread.interrupt();
				experienceUploadThread.join();
			}
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
	}
	
	private Tensor action = null;
	
	private Tensor selectActionFromObservation(Tensor state, long i) {
		Tensor out = nn.forward(state);
		return actionStrategy.selectActionFromOutput(out, i);
	}
	
	private class AgentRunnable implements Runnable {

		@Override
		public void run() {
			env.reset();
			Tensor observation = env.getObservation();

			for(long i = 0; acting; i++) {
				if(syncInterval > 0 && i % syncInterval == 0){
					// sync parameters
					try {
						nn.loadParameters(tag);
					} catch(Exception e){
						System.out.println("Failed to load parameters with tag "+tag);
					}
				}
				
				Tensor action = selectActionFromObservation(observation, i);

				float reward = env.performAction(action);
				Tensor nextObservation = env.getObservation();

				synchronized(samples){
					samples.add(new ExperiencePoolSample(observation, action, reward, nextObservation));
					if(i % experienceInterval == 0){
						samples.notifyAll();
					}
				}

				observation = nextObservation;
				// if nextObservation was null, this is a terminal state - reset environment and start over
				if(env.getObservation() == null){
					env.reset();
					observation = env.getObservation();
				}

				if(gcInterval > 0 && i % gcInterval == 0){
					System.gc();
				}
			}
		}
	}
	
	private class UploadRunnable implements Runnable {

		@Override
		public void run() {
			if(clean){
				if(pool!=null)
					pool.reset();
			}
			
			if(pool!=null){
				pool.setMaxSize(experienceSize);
			}
			
			while(acting){
				List<ExperiencePoolSample> toUpdate = new ArrayList<>();
				synchronized(samples){
					if(samples.size() >= experienceInterval){
						toUpdate.addAll(samples);
						samples.clear();
					} else {
						try {
							samples.wait();
						} catch (InterruptedException e) {
						}
					}
				}
				
				if(pool!=null && !toUpdate.isEmpty()){
					pool.addSamples(toUpdate);
				}
			}
		}
		
	}
}
