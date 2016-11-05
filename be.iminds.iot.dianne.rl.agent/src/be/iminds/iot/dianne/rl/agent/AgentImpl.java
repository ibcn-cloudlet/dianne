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
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Dictionary;
import java.util.HashMap;
import java.util.Hashtable;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.osgi.framework.BundleContext;
import org.osgi.framework.Constants;
import org.osgi.framework.ServiceRegistration;
import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Deactivate;
import org.osgi.service.component.annotations.Reference;
import org.osgi.service.component.annotations.ReferenceCardinality;
import org.osgi.service.component.annotations.ReferencePolicy;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.dataset.DianneDatasets;
import be.iminds.iot.dianne.api.nn.Dianne;
import be.iminds.iot.dianne.api.nn.NeuralNetwork;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkInstanceDTO;
import be.iminds.iot.dianne.api.nn.util.StrategyFactory;
import be.iminds.iot.dianne.api.repository.RepositoryListener;
import be.iminds.iot.dianne.api.rl.agent.ActionStrategy;
import be.iminds.iot.dianne.api.rl.agent.Agent;
import be.iminds.iot.dianne.api.rl.agent.AgentListener;
import be.iminds.iot.dianne.api.rl.agent.AgentProgress;
import be.iminds.iot.dianne.api.rl.dataset.ExperiencePool;
import be.iminds.iot.dianne.api.rl.dataset.ExperiencePoolSample;
import be.iminds.iot.dianne.api.rl.environment.Environment;
import be.iminds.iot.dianne.nn.util.DianneConfigHandler;
import be.iminds.iot.dianne.rl.agent.config.AgentConfig;

@Component(property={"aiolos.unique=true"})
public class AgentImpl implements Agent {

	private UUID agentId;
	
	private ExecutorService listenerExecutor = Executors.newSingleThreadExecutor(); 
	private List<AgentListener> listeners = Collections.synchronizedList(new ArrayList<>());
	private volatile boolean wait = false;
	
	private Map<String, Environment> envs = new HashMap<String, Environment>();
	private Dianne dianne;

	private NeuralNetwork[] nns;
	private DianneDatasets datasets;
	
	private ExperiencePool pool;
	private Environment env;
	
	private AgentConfig config;
	private Map<String, String> properties;
	
	private Thread actingThread;
	private long i = 0;
	private volatile boolean acting;
	
	private ActionStrategy strategy;
	private StrategyFactory<ActionStrategy> factory;
	private volatile AgentProgress progress;
	
	// separate thread for updating the experience pool
	private Thread experienceUploadThread;
	private List<ExperiencePoolSample> storeBuffer;
	private List<ExperiencePoolSample> uploadBuffer;
	private List<ExperiencePoolSample> upload; 
	private volatile boolean bufferReady = false;
	private volatile boolean uploading = false;
	private int count = 0;
	
	// repository listener to sync with repo
	private BundleContext context;
	private ServiceRegistration<RepositoryListener> reg;
	private volatile boolean sync = false;
	
	
	@Reference
	void setDianne(Dianne d){
		dianne = d;
	}
	
	@Reference
	void setDianneDatasets(DianneDatasets d){
		datasets = d;
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

	@Reference
	void setActionFactoryStrategy(StrategyFactory<ActionStrategy> f){
		this.factory = f;
	}
	
	@Activate
	public void activate(BundleContext context){
		this.context = context;
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
	public void act(String environment, String experiencePool, Map<String, String> config, NeuralNetworkInstanceDTO... nni)
			throws Exception {
		synchronized(this){
			if (acting)
				throw new Exception("Already running an Agent here");

			acting = true;
		}
		
		try {
			System.out.println("Agent Configuration");
			System.out.println("===================");
	
			this.config = DianneConfigHandler.getConfig(config, AgentConfig.class);
			this.properties = config;
			
			strategy = factory.create(this.config.strategy);
			if(strategy==null){
				acting = false;
				throw new RuntimeException("Invalid strategy selected: "+strategy);
			}
			
			if(nni == null){
				// for Agents a null is allowed, e-g when using hard-coded policies
				nns = new NeuralNetwork[0];
			} else {
				System.out.println("Neural Network(s)");
				System.out.println("---");
				
				int n = 0;
				nns = new NeuralNetwork[nni.length];
				for(NeuralNetworkInstanceDTO dto : nni){
					if(dto != null){
						NeuralNetwork nn = dianne.getNeuralNetwork(dto).getValue();
						nns[n++] = nn;
						System.out.println("* "+dto.name);
					}
				}
				System.out.println("---");
			}
			
			// setup environment
			env = envs.get(environment);
			if(env==null){
				throw new RuntimeException("Environment "+environment+" does not exist");
			}
			env.setup(properties);
			
			if(experiencePool != null){
				// add env state/actionDims in case we need to construct xp pool
				if(!config.containsKey("stateDims")){
					int[] stateDims = env.observationDims();
					String sd = Arrays.toString(stateDims);
					config.put("stateDims", sd.substring(1, sd.length()-1));
				}
				if(!config.containsKey("actionDims")){
					int[] actionDims = env.actionDims();
					String ad = Arrays.toString(actionDims);
					config.put("actionDims", ad.substring(1, ad.length()-1));
				}
				
				Dataset d = datasets.configureDataset(experiencePool, config);
				if(d == null || !(d instanceof ExperiencePool)){
					acting = false;
					throw new RuntimeException("Invalid experience pool: "+experiencePool);
				}
				pool = (ExperiencePool) d;
			}
	
			storeBuffer = new ArrayList<>();
			uploadBuffer = new ArrayList<>();
			
			actingThread = new Thread(new AgentRunnable());
			experienceUploadThread = new Thread(new UploadRunnable());
			actingThread.start();
			experienceUploadThread.start();
		
		} catch(Exception e){
			System.err.println("Failed starting agent");
			e.printStackTrace();
			acting = false;
			throw e;
		}
	}

	@Override
	public AgentProgress getProgress() {
		return progress;
	}
	
	@Override
	public synchronized void stop() {
		if(!acting)
			return;
		
		try {
			if (actingThread != null && actingThread.isAlive()) {
				acting = false;
				actingThread.interrupt();
				actingThread.join();
				experienceUploadThread.interrupt();
				experienceUploadThread.join();
			}
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
	}
	
	private class AgentRunnable implements Runnable {

		@Override
		public void run() {
			ExperiencePoolSample s = new ExperiencePoolSample();
			
			try {
				// setup repo listener
				Dictionary<String, Object> props = new Hashtable<>();
				String[] t = new String[]{":"+config.tag};
				props.put("targets", t);
				props.put("aiolos.unique", true);
				reg = context.registerService(RepositoryListener.class, new RepositoryListener() {
					@Override
					public void onParametersUpdate(UUID nnId, Collection<UUID> moduleIds, String... tag) {
						sync = true;
					}
				}, props);
				
				// make sure to sync initially
				sync = true;
				
				// setup action strategy
				strategy.setup(properties, env, nns);
		
				s.input = env.getObservation(s.input);
	
				for(i = 0; acting; i++) {
					// sync parameters
					if(sync){
						for(int k=0;k<nns.length;k++){
							try {
								nns[k].loadParameters(config.tag);
							} catch(Exception e){
								System.out.println("Failed loading parameters for nn "+nns[k].getId());
							}
						}
						sync = false;
					}
					
					progress = strategy.processIteration(i, s.input);
					s.target = progress.action;
	
					s.reward= env.performAction(s.target);
					progress.reward = s.reward;
					
					s.nextState = env.getObservation(s.nextState);
					s.isTerminal = s.nextState == null;
	
					if(config.trace && i % config.traceInterval == 0){
						System.out.println(progress);
					}
					
					// upload in batch
					if(pool != null) {
						ExperiencePoolSample b;
						if(storeBuffer.size() <= count){
							b = new ExperiencePoolSample();
							storeBuffer.add(b);
						} else {
							b = storeBuffer.get(count);
						}
						b.input = s.input.copyInto(b.input);
						b.target = s.target.copyInto(b.target);
						b.reward = s.reward;
						b.isTerminal = s.isTerminal;
						b.nextState = s.isTerminal ? null : s.nextState.copyInto(b.nextState);
						count++;
						
						if(b.isTerminal || count % config.experienceInterval == 0){
							System.out.println("PREPARE UPLOAD!");
							// sequence finished, switch to upload
							// check if upload finished
							// if still uploading ... wait now
							synchronized(uploadBuffer){
								while(uploading){
									try {
										uploadBuffer.wait();
									} catch (InterruptedException e) {
									}
								}
							}

							List<ExperiencePoolSample> temp = uploadBuffer;
							uploadBuffer = storeBuffer;
							storeBuffer = temp;
							upload = uploadBuffer.subList(0, count);
							
							
							count = 0;
							
							// Need to notify on previous buffer = current upload
							synchronized(uploadBuffer){
								bufferReady = true;
								uploadBuffer.notifyAll();
							}
						}
					}
	
					// if this is a terminal state - reset environment and start over
					if(s.isTerminal){
						env.reset();
						s.input = env.getObservation(s.input);
					} else {
						s.input = s.nextState.copyInto(s.input);
					}
					
					publishProgress(progress);
				}
			} catch(Throwable t){
				if(t.getCause() != null && t.getCause() instanceof InterruptedException){
					return;
				}
				
				acting = false;
				
				System.out.println("Error during acting");
				t.printStackTrace();
				
				publishError(t);
			} finally {
				env.cleanup();
				
				datasets.releaseDataset(pool);
				
				if(reg != null){
					reg.unregister();
				}
				
				acting = false;
				
				publishDone();
			}
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
				synchronized(storeBuffer){
					while(!bufferReady){
						try {
							storeBuffer.wait();
						} catch (InterruptedException e) {
							if(!acting)
								return;
							else
								e.printStackTrace();
						}
					}
				}

				bufferReady = false;
				uploading = true;
				
				System.out.println("UPLOADING "+upload.size()+" SAMPLES!");
				if(pool!=null){
					pool.addSequence(upload);
				}
				
				synchronized(uploadBuffer){
					uploading = false;
					uploadBuffer.notifyAll();
				}
				
			}
		}
		
	}
	
	private void publishProgress(final AgentProgress progress){
		if(!acting)
			return;
		
		synchronized(listenerExecutor){
			if(wait){
				try {
					listenerExecutor.wait();
				} catch (InterruptedException e) {
					wait = false;
					return;
				}
			}
			wait = true;
		}
		
		listenerExecutor.submit(()->{
			List<AgentListener> copy = new ArrayList<>();
			synchronized(listeners){
				copy.addAll(listeners);
			}
			for(AgentListener l : copy){
				l.onProgress(agentId, progress);
			}
			
			synchronized(listenerExecutor){
				wait = false;
				listenerExecutor.notifyAll();
			}
		});
	}
	
	private void publishError(final Throwable t){
		List<AgentListener> copy = new ArrayList<>();
		synchronized(listeners){
			copy.addAll(listeners);
		}
		for(AgentListener l : copy){
			l.onException(agentId, t.getCause()!=null ? t.getCause() : t);
		}
	}
	
	private void publishDone(){
		List<AgentListener> copy = new ArrayList<>();
		synchronized(listeners){
			copy.addAll(listeners);
		}
		for(AgentListener l : copy){
			l.onFinish(agentId);
		}
	}
	
	@Reference(cardinality=ReferenceCardinality.MULTIPLE, 
			policy=ReferencePolicy.DYNAMIC)
	void addListener(AgentListener listener, Map<String, Object> properties){
		String[] targets = (String[])properties.get("targets");
		if(targets!=null){
			boolean listen = false;
			for(String target : targets){
				if(agentId.toString().equals(target)){
					listen  = true;
				}
			}
			if(!listen)
				return;	
		}
		this.listeners.add(listener);
	}
	
	void removeListener(AgentListener listener, Map<String, Object> properties){
		this.listeners.remove(listener);
	}
}
