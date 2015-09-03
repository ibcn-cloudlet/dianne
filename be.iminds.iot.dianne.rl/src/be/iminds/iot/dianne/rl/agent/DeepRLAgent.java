package be.iminds.iot.dianne.rl.agent;

import java.util.Arrays;
import java.util.Collection;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Deactivate;
import org.osgi.service.component.annotations.Reference;
import org.osgi.service.component.annotations.ReferenceCardinality;
import org.osgi.service.component.annotations.ReferencePolicy;

import be.iminds.iot.dianne.api.nn.module.ForwardListener;
import be.iminds.iot.dianne.api.nn.module.Input;
import be.iminds.iot.dianne.api.nn.module.Module;
import be.iminds.iot.dianne.api.nn.module.Output;
import be.iminds.iot.dianne.api.nn.module.Trainable;
import be.iminds.iot.dianne.api.nn.module.Module.Mode;
import be.iminds.iot.dianne.api.nn.module.dto.ModuleInstanceDTO;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkDTO;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkInstanceDTO;
import be.iminds.iot.dianne.api.nn.runtime.ModuleManager;
import be.iminds.iot.dianne.api.repository.DianneRepository;
import be.iminds.iot.dianne.api.repository.RepositoryListener;
import be.iminds.iot.dianne.api.rl.Agent;
import be.iminds.iot.dianne.api.rl.Environment;
import be.iminds.iot.dianne.api.rl.ExperiencePool;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;

@Component
public class DeepRLAgent implements Agent, RepositoryListener, ForwardListener {

	private TensorFactory factory;
	private ModuleManager runtime;
	private DianneRepository repository;
	private Map<String, ExperiencePool> pools = new HashMap<String, ExperiencePool>();
	private Map<String, Environment> envs = new HashMap<String, Environment>();
	private Map<String, ActionStrategy> strategies = new HashMap<String, ActionStrategy>();

	
	private NeuralNetworkInstanceDTO nni;
	private Input input;
	private Output output;
	
	private Tensor nnOutput;

	private Thread actingThread;
	private volatile boolean update;
	private volatile boolean acting;

	private String tag = "run";
	
	private ActionStrategy actionStrategy;
	
	@Reference
	void setTensorFactory(TensorFactory factory) {
		this.factory = factory;
	}
	
	@Reference
	void setModuleManager(ModuleManager runtime){
		this.runtime = runtime;
	}
	
	@Reference
	void setDianneRepository(DianneRepository repository){
		this.repository = repository;
	}

	@Reference(cardinality = ReferenceCardinality.MULTIPLE, policy = ReferencePolicy.DYNAMIC)
	public void addExperiencePool(ExperiencePool pool, Map<String, Object> properties) {
		String name = (String) properties.get("name");
		this.pools.put(name, pool);
	}

	public void removeExperiencePool(ExperiencePool pool, Map<String, Object> properties) {
		String name = (String) properties.get("name");
		this.pools.remove(name);
	}

	@Reference(cardinality = ReferenceCardinality.MULTIPLE, policy = ReferencePolicy.DYNAMIC)
	public void addEnvironment(Environment env, Map<String, Object> properties) {
		String name = (String) properties.get("name");
		this.envs.put(name, env);
	}

	public void removeEnvironment(Environment env, Map<String, Object> properties) {
		String name = (String) properties.get("name");
		this.envs.remove(name);
	}

	@Reference(cardinality = ReferenceCardinality.MULTIPLE, policy = ReferencePolicy.DYNAMIC)
	public void addStrategy(ActionStrategy s, Map<String, Object> properties) {
		String strategy = (String) properties.get("strategy");
		this.strategies.put(strategy, s);
	}

	public void removeStrategy(ActionStrategy s, Map<String, Object> properties) {
		String strategy = (String) properties.get("strategy");
		this.strategies.remove(strategy);
	}
	
	@Deactivate
	void deactivate() {
		if(acting)
			stop();
	}

	@Override
	public synchronized void act(String nnName, String environment, String experiencePool, Map<String, String> config)
			throws Exception {
		if (acting)
			throw new Exception("Already running an Agent here");
		else if (nnName == null || !repository.availableNeuralNetworks().contains(nnName))
			throw new Exception("Network name " + nnName + " is null or not available");
		else if (environment == null || !envs.containsKey(environment))
			throw new Exception("Environment " + environment + " is null or not available");
		else if (experiencePool != null && !pools.containsKey(experiencePool))
			throw new Exception("ExperiencePool " + experiencePool + " is not available");

		if(config.containsKey("tag"))
			tag = config.get("tag"); 
		
		String strategy = "greedy";
		if(config.containsKey("strategy"))
			strategy = config.get("strategy");
		
		actionStrategy = strategies.get(strategy);
		if(actionStrategy==null)
			throw new RuntimeException("Invalid strategy selected: "+strategy);
		
		actionStrategy.configure(config);
		
		NeuralNetworkDTO nn = repository.loadNeuralNetwork(nnName);
		
		UUID nnId = UUID.randomUUID();
		List<ModuleInstanceDTO> moduleInstances = nn.modules.stream().map(m -> runtime.deployModule(m, nnId)).collect(Collectors.toList());
		nni = new NeuralNetworkInstanceDTO(nnId, nnName, moduleInstances);
		
		Supplier<Stream<Module>> modules = () -> nni.modules.stream().map(mi -> runtime.getModule(mi.moduleId, mi.nnId));
		input = (Input) modules.get().filter(m -> m instanceof Input).findAny().get();
		output = (Output) modules.get().filter(m -> m instanceof Output).findAny().get();
		
		input.setMode(EnumSet.of(Mode.BLOCKING));
		output.addForwardListener(this);
		
		loadParameters();
		
		Environment env = envs.get(environment);
		ExperiencePool pool = pools.get(experiencePool);

		actingThread = new Thread(new AgentRunnable(env, pool));
		acting = true;
		actingThread.start();
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
	
	@Override
	public void onParametersUpdate(Collection<UUID> moduleIds, String... tag) {
		// TODO should be done by a targets service property? Do via config
		if(nni.modules.stream().anyMatch(m -> moduleIds.contains(m.moduleId))
				&& Arrays.stream(tag).anyMatch(t -> t.equals(this.tag))) {
			update = true;
		}
	}
	
	private void loadParameters(){
		System.out.println("Agent loading parameters for "+nni.name+" "+tag);
		Map<UUID, Tensor> parameters = repository.loadParameters(nni.name, tag);
		parameters.entrySet().stream().forEach(e -> {
			Trainable module = (Trainable) runtime.getModule(e.getKey(), nni.id);
			module.setParameters(e.getValue());
		});
	}
	
	private Tensor selectActionFromObservation(Tensor state, long i) {
		if(update) {
			loadParameters();
			update = false;
		}
		
		synchronized(this) {
			input.input(state);
			
			try {
				wait();
			} catch (InterruptedException e) {}
		}
		
		return actionStrategy.selectActionFromOutput(nnOutput, i);
	}
	
	@Override
	public void onForward(Tensor output, String... tags) {
		synchronized(this) {
			nnOutput = output;
			notify();
		}
	}
	
	private class AgentRunnable implements Runnable {

		private final Environment env;
		private final ExperiencePool pool;
		
		public AgentRunnable(Environment env, ExperiencePool pool) {
			this.env = env;
			this.pool = pool;
		}

		@Override
		public void run() {
			Tensor observation = env.getObservation();

			for(long i = 0; acting; i++) {
				Tensor action = selectActionFromObservation(observation, i);

				float reward = env.performAction(action);
				Tensor nextObservation = env.getObservation();

				if (pool != null)
					pool.addSample(observation, action, reward, nextObservation);

				observation = nextObservation;
			}
		}
	}
}
