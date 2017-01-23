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
import be.iminds.iot.dianne.api.dataset.Sequence;
import be.iminds.iot.dianne.api.nn.Dianne;
import be.iminds.iot.dianne.api.nn.NeuralNetwork;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkInstanceDTO;
import be.iminds.iot.dianne.api.nn.util.StrategyFactory;
import be.iminds.iot.dianne.api.repository.RepositoryListener;
import be.iminds.iot.dianne.api.rl.agent.ActionController;
import be.iminds.iot.dianne.api.rl.agent.ActionStrategy;
import be.iminds.iot.dianne.api.rl.agent.Agent;
import be.iminds.iot.dianne.api.rl.agent.AgentListener;
import be.iminds.iot.dianne.api.rl.agent.AgentProgress;
import be.iminds.iot.dianne.api.rl.dataset.ExperiencePool;
import be.iminds.iot.dianne.api.rl.dataset.ExperiencePoolSample;
import be.iminds.iot.dianne.api.rl.environment.Environment;
import be.iminds.iot.dianne.nn.util.DianneConfigHandler;
import be.iminds.iot.dianne.rl.agent.config.AgentConfig;
import be.iminds.iot.dianne.tensor.Tensor;

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
	private String environment;
	private Environment env;
	
	private AgentConfig config;
	private Map<String, String> properties;
	
	private Thread actingThread;
	private long i = 0;
	private long seq = 0;
	private long episode = 0;
	private volatile boolean acting;
	
	private ActionStrategy strategy;
	private StrategyFactory<ActionStrategy> factory;
	private volatile AgentProgress progress;
	
	private List<ExperiencePoolSample> uploadBuffer;
	private Sequence<ExperiencePoolSample> upload; 
	private int count = 0;
	
	// repository listener to sync with repo
	private BundleContext context;
	private ServiceRegistration<RepositoryListener> repoListenerReg;
	private volatile boolean sync = false;
	
	// in case of manual action strategy
	private ServiceRegistration<ActionController> actionListenerReg;
	
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
			this.environment = environment;
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
	
			uploadBuffer = new ArrayList<>();
			
			actingThread = new Thread(new AgentRunnable());
			actingThread.start();
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
			acting = false;

			if (actingThread != null && actingThread.isAlive()) {
				actingThread.interrupt();
				actingThread.join();
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
				repoListenerReg = context.registerService(RepositoryListener.class, new RepositoryListener() {
					@Override
					public synchronized void onParametersUpdate(UUID nnId, Collection<UUID> moduleIds, String... tag) {
						if(sync == false){
							sync = true;
							episode++;
						}
					}
				}, props);
				
				// make sure to sync initially
				sync = true;
				
				// set count to zero
				count = 0;
				seq = 0;
				episode = 0;
				
				// setup action strategy
				strategy.setup(properties, env, nns);
				// this allows the strategy to adapt config in setup
				config = DianneConfigHandler.getConfig(properties, AgentConfig.class);
				
				// TODO this his hard coded for ManualActionStrategy ... have something better?
				if(strategy instanceof ActionController){
					props.put("environment", environment);
					actionListenerReg = context.registerService(ActionController.class, (ActionController)strategy, props);
				}
		
				s.input = env.getObservation(s.input);
	
				progress = new AgentProgress(seq, 0, 0, episode);
				
				if(config.clear){
					pool.reset();
				}
				
				while(acting) {
					// sync parameters
					if(sync && count == 0){
						for(int k=0;k<nns.length;k++){
							try {
								nns[k].loadParameters(config.tag);
							} catch(Exception e){
								System.out.println("Failed loading parameters for nn "+nns[k].getId());
							}
						}
						sync = false;
					}
					
					// select action according to strategy
					s.target = strategy.processIteration(progress.sequence, progress.iterations, s.input);
	
					// execute action and get reward
					float reward = env.performAction(s.target);
					if(s.reward == null){
						s.reward = new Tensor(1);
					}
					s.reward.set(reward, 0);
					
					// get the next state
					s.nextState = env.getObservation(s.nextState);
					
					// check if terminal
					if(s.terminal == null){
						s.terminal = new Tensor(1);
					}
					s.terminal.set(s.nextState == null ? 0.0f : 1.0f, 0);
					
					// update progress
					progress.reward+=reward;
					progress.iterations++;
					
					// upload in batch
					if(pool != null) {
						ExperiencePoolSample b;
						if(uploadBuffer.size() <= count){
							b = new ExperiencePoolSample();
							uploadBuffer.add(b);
						} else {
							b = uploadBuffer.get(count);
						}
						b.input = s.input.copyInto(b.input);
						b.target = s.target.copyInto(b.target);
						b.reward = s.reward.copyInto(b.reward);
						b.terminal = s.terminal.copyInto(b.terminal);
						b.nextState = s.isTerminal() ? null : s.nextState.copyInto(b.nextState);
						count++;
						
						if(b.isTerminal()){
							// sequence finished, upload to pool
							upload = new Sequence<ExperiencePoolSample>(uploadBuffer.subList(0, count), count);
							if(pool!=null){
								try {
									pool.addSequence(upload);
								} catch(Exception e){
									System.out.println("Failed to upload to experience pool "+e.getMessage());
								}
							}
							count = 0;
						}
					}
	
					// if this is a terminal state - reset environment and start over
					// TODO what with infinite horizon environments?
					if(s.isTerminal()){
						publishProgress(progress);
						
						// trace agent per sequence
						if(config.trace && seq % config.traceInterval == 0){
							System.out.println(progress);
						}
						
						seq++;
						progress = new AgentProgress(seq, 0, 0, episode);
						
						do {
							env.reset();
							s.input = env.getObservation(s.input);
							if(s.input==null){
								System.out.println("Observation null after reset, trying to reinitialize environment.");
							}
						} while(s.input == null);
					} else {
						s.input = s.nextState.copyInto(s.input);
					}
				}
			} catch(Throwable t){
				if(t.getCause() != null && t.getCause() instanceof InterruptedException){
					return;
				}
				
				acting = false;
				
				t.printStackTrace();
				
				publishError(t);
			} finally {
				env.cleanup();
				
				datasets.releaseDataset(pool);
				
				if(repoListenerReg != null){
					repoListenerReg.unregister();
				}
				
				if(actionListenerReg != null){
					actionListenerReg.unregister();
				}
				
				acting = false;
				
				publishDone();
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
