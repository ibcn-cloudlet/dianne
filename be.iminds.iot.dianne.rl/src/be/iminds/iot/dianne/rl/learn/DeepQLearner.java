package be.iminds.iot.dianne.rl.learn;

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

import be.iminds.iot.dianne.api.nn.learn.Learner;
import be.iminds.iot.dianne.api.nn.learn.Processor;
import be.iminds.iot.dianne.api.nn.module.Input;
import be.iminds.iot.dianne.api.nn.module.Module;
import be.iminds.iot.dianne.api.nn.module.Output;
import be.iminds.iot.dianne.api.nn.module.Trainable;
import be.iminds.iot.dianne.api.nn.module.dto.ModuleInstanceDTO;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkDTO;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkInstanceDTO;
import be.iminds.iot.dianne.api.nn.runtime.ModuleManager;
import be.iminds.iot.dianne.api.repository.DianneRepository;
import be.iminds.iot.dianne.api.rl.ExperiencePool;
import be.iminds.iot.dianne.nn.learn.processors.MomentumProcessor;
import be.iminds.iot.dianne.nn.learn.processors.RegularizationProcessor;
import be.iminds.iot.dianne.rl.learn.processors.TimeDifferenceProcessor;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;

@Component
public class DeepQLearner implements Learner {

	private TensorFactory factory;
	private ModuleManager runtime;
	private DianneRepository repository;
	private Map<String, ExperiencePool> pools = new HashMap<String, ExperiencePool>();

	private NeuralNetworkInstanceDTO nni;
	private Input input;
	private Output output;
	private Map<UUID, Trainable> toTrain;

	private NeuralNetworkInstanceDTO targetNni;
	private Input targetInput;
	private Output targetOutput;
	private Map<UUID, Trainable> targetToTrain;

	private Thread learningThread;
	private volatile boolean learning;

	private String tag = "learn";
	private int updateInterval = 10000;

	@Reference
	void setTensorFactory(TensorFactory factory) {
		this.factory = factory;
	}

	@Reference
	void setModuleManager(ModuleManager runtime) {
		this.runtime = runtime;
	}

	@Reference
	void setDianneRepository(DianneRepository repository) {
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

	@Deactivate
	void deactivate() {
		if (learning)
			stop();
	}

	@Override
	public void learn(String nnName, String experiencePool, Map<String, String> config) throws Exception {
		if (learning)
			throw new Exception("Already running a Learner here");
		else if (nnName == null || !repository.availableNeuralNetworks().contains(nnName))
			throw new Exception("Network name " + nnName + " is null or not available");
		else if (experiencePool == null || !pools.containsKey(experiencePool))
			throw new Exception("ExperiencePool " + experiencePool + " is null or not available");

		if (config.containsKey("tag"))
			tag = config.get("tag");

		if (config.containsKey("updateInterval"))
			updateInterval = Integer.parseInt(config.get("updateInterval"));

		NeuralNetworkDTO nn = repository.loadNeuralNetwork(nnName);

		UUID nnId = UUID.randomUUID();
		List<ModuleInstanceDTO> moduleInstances = nn.modules.stream().map(m -> runtime.deployModule(m, nnId)).collect(Collectors.toList());
		nni = new NeuralNetworkInstanceDTO(nnId, nnName, moduleInstances);

		UUID targetNnId = UUID.randomUUID();
		List<ModuleInstanceDTO> targetModuleInstances = nn.modules.stream().map(m -> runtime.deployModule(m, targetNnId)).collect(Collectors.toList());
		targetNni = new NeuralNetworkInstanceDTO(targetNnId, nnName, targetModuleInstances);

		Supplier<Stream<Module>> modules = () -> nni.modules.stream().map(mi -> runtime.getModule(mi.moduleId, mi.nnId));
		input = (Input) modules.get().filter(m -> m instanceof Input).findAny().get();
		output = (Output) modules.get().filter(m -> m instanceof Output).findAny().get();
		toTrain = modules.get().filter(m -> m instanceof Trainable).collect(Collectors.toMap(m -> m.getId(), m -> (Trainable) m));

		Supplier<Stream<Module>> targetModules = () -> targetNni.modules.stream().map(mi -> runtime.getModule(mi.moduleId, mi.nnId));
		targetInput = (Input) targetModules.get().filter(m -> m instanceof Input).findAny().get();
		targetOutput = (Output) targetModules.get().filter(m -> m instanceof Output).findAny().get();
		targetToTrain = modules.get().filter(m -> m instanceof Trainable).collect(Collectors.toMap(m -> m.getId(), m -> (Trainable) m));
		
		loadParameters();
		
		ExperiencePool pool = pools.get(experiencePool);

		Processor p = new MomentumProcessor(new RegularizationProcessor(new TimeDifferenceProcessor(factory, input, output, toTrain, targetInput, targetOutput, pool, config)));

		learningThread = new Thread(new DeepQLearnerRunnable(p));
		learning = true;
		learningThread.start();
	}

	@Override
	public void stop() {
		try {
			if (learningThread != null && learningThread.isAlive()) {
				learning = false;
				learningThread.join();
			}
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
	}

	private void loadParameters() {
		try {
			Map<UUID, Tensor> parameters = repository.loadParameters(toTrain.keySet(), tag);
			parameters.entrySet().stream().forEach(e -> {
				toTrain.get(e.getKey()).setParameters(e.getValue());
				targetToTrain.get(e.getKey()).setParameters(e.getValue());
			});
		} catch (Exception ex) {
			Map<UUID, Tensor> parameters = toTrain.entrySet().stream()
					.collect(Collectors.toMap(e -> e.getKey(), e -> e.getValue().getParameters()));
			repository.storeParameters(parameters, tag);
			
			toTrain.entrySet().stream().forEach(e -> targetToTrain.get(e.getKey()).setParameters(e.getValue().getParameters()));
		}
	}

	private void publishParameters() {
		System.out.println("Publish parameters "+tag);
		Map<UUID, Tensor> deltaParameters = toTrain.entrySet().stream()
				.collect(Collectors.toMap(e -> e.getKey(), e -> factory.getTensorMath().sub(null,
						e.getValue().getParameters(), targetToTrain.get(e.getKey()).getParameters())));

		repository.accParameters(deltaParameters, tag);
	}

	private class DeepQLearnerRunnable implements Runnable {

		private final Processor p;

		public DeepQLearnerRunnable(Processor p) {
			this.p = p;
		}

		@Override
		public void run() {
			double error = 0, runningAvg = 0;

			for (long i = 1; learning; i++) {
				toTrain.values().stream().forEach(Trainable::zeroDeltaParameters);
				
				error = p.processNext();
				runningAvg = 0.99f * runningAvg + 0.01f * error;

				if(i % 1000 == 0)
					System.out.println(i+"\terror: "+error + "\trunning avg: " + runningAvg / i);

				toTrain.values().stream().forEach(Trainable::updateParameters);

				if (updateInterval > 0 && i % updateInterval == 0)
					publishParameters();
			}

			publishParameters();
		}

	}
}
