package be.iminds.iot.dianne.rl.agent;

import java.util.ArrayList;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Deactivate;
import org.osgi.service.component.annotations.Reference;
import org.osgi.service.component.annotations.ReferenceCardinality;
import org.osgi.service.component.annotations.ReferencePolicy;

import be.iminds.iot.dianne.api.nn.Dianne;
import be.iminds.iot.dianne.api.nn.NeuralNetwork;
import be.iminds.iot.dianne.api.nn.module.Module.Mode;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkInstanceDTO;
import be.iminds.iot.dianne.api.rl.Agent;
import be.iminds.iot.dianne.api.rl.Environment;
import be.iminds.iot.dianne.api.rl.ExperiencePool;
import be.iminds.iot.dianne.api.rl.ExperiencePoolSample;
import be.iminds.iot.dianne.rl.agent.strategy.ActionStrategy;
import be.iminds.iot.dianne.tensor.Tensor;

@Component
public class DeepRLAgent implements Agent {

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
	
	@Deactivate
	void deactivate() {
		if(acting)
			stop();
	}

	@Override
	public synchronized void act(NeuralNetworkInstanceDTO nni, String environment, String experiencePool, Map<String, String> config)
			throws Exception {
		if (acting)
			throw new Exception("Already running an Agent here");
		else if (environment == null || !envs.containsKey(environment))
			throw new Exception("Environment " + environment + " is null or not available");
		else if (experiencePool != null && !pools.containsKey(experiencePool))
			throw new Exception("ExperiencePool " + experiencePool + " is not available");

		if(config.containsKey("tag"))
			tag = config.get("tag"); 
		
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
		System.out.println("* experienceInterval = "+experienceInterval);
		System.out.println("* experienceSize = "+experienceSize);
		System.out.println("---");
		
		actionStrategy = strategies.get(strategy);
		if(actionStrategy==null)
			throw new RuntimeException("Invalid strategy selected: "+strategy);
		
		actionStrategy.configure(config);
		
		nn = dianne.getNeuralNetwork(nni);
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
					samples.notifyAll();
				}

				observation = nextObservation;
				// if nextObservation was null, this is a terminal state - reset environment and start over
				if(observation == null){
					env.reset();
					observation = env.getObservation();
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
					if(samples.size() > experienceInterval){
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
