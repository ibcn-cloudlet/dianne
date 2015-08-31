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
public class GreedyDeepQAgent implements Agent, RepositoryListener {

	private TensorFactory factory;
	private ModuleManager runtime;
	private DianneRepository repository;
	private Map<String, ExperiencePool> pools = new HashMap<String, ExperiencePool>();
	private Map<String, Environment> envs = new HashMap<String, Environment>();
	
	private NeuralNetworkInstanceDTO nni;

	private Thread actingThread;
	private volatile boolean update;
	private volatile boolean acting;

	private String tag = "run";
	private float epsilon = 0.1f;

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
		
		if (config.containsKey("epsilon"))
			epsilon = Float.parseFloat(config.get("epsilon"));
		
		NeuralNetworkDTO nn = repository.loadNeuralNetwork(nnName);
		
		UUID nnId = UUID.randomUUID();
		List<ModuleInstanceDTO> moduleInstances = nn.modules.stream().map(m -> runtime.deployModule(m, nnId)).collect(Collectors.toList());
		nni = new NeuralNetworkInstanceDTO(nnId, nnName, moduleInstances);
		
		Supplier<Stream<Module>> modules = () -> nni.modules.stream().map(mi -> runtime.getModule(mi.moduleId, mi.nnId));
		Input input = (Input) modules.get().filter(m -> m instanceof Input).findAny().get();
		Output output = (Output) modules.get().filter(m -> m instanceof Output).findAny().get();
		
		loadParameters();
		
		Environment env = envs.get(environment);
		ExperiencePool pool = pools.get(experiencePool);

		actingThread = new Thread(new GreedyDeepQAgentRunnable(input, output, env, pool, epsilon));
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
		if(nni.modules.stream().anyMatch(m -> moduleIds.contains(m.moduleId))
				&& Arrays.stream(tag).anyMatch(t -> t.equals(this.tag))) {
			update = true;
		}
	}
	
	private void loadParameters(){
		Map<UUID, Tensor> parameters = repository.loadParameters(nni.name, tag);
		parameters.entrySet().stream().forEach(e -> {
			Trainable module = (Trainable) runtime.getModule(e.getKey(), nni.id);
			module.setParameters(e.getValue());
		});
	}

	private class GreedyDeepQAgentRunnable implements Runnable, ForwardListener {

		private final Input input;
		private final Output output;
		private final Environment env;
		private final ExperiencePool pool;
		private final float epsilon;
		
		private Tensor q;

		public GreedyDeepQAgentRunnable(Input input, Output output, Environment env, ExperiencePool pool, float epsilon) {
			this.input = input;
			this.output = output;
			this.env = env;
			this.pool = pool;
			this.epsilon = epsilon;
			
			this.input.setMode(EnumSet.of(Mode.BLOCKING));
			this.output.addForwardListener(this);
		}

		@Override
		public void run() {
			Tensor state = env.getObservation();

			while (acting) {
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
				
				Tensor action = factory.createTensor(q.size());
				action.fill(-1);

				if (Math.random() < epsilon) {
					action.set(1, (int) (Math.random() * action.size()));
				} else {
					action.set(1, factory.getTensorMath().argmax(q));
				}

				float reward = env.performAction(action);
				Tensor nextState = env.getObservation();

				if (pool != null)
					pool.addSample(state, action, reward, nextState);

				state = nextState;
			}
		}

		@Override
		public void onForward(Tensor output, String... tags) {
			synchronized(this) {
				q = output;
				notify();
			}
		}
	}
}
