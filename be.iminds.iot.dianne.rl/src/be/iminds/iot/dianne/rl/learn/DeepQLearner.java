package be.iminds.iot.dianne.rl.learn;

import java.util.ArrayList;
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

import be.iminds.iot.dianne.api.log.DataLogger;
import be.iminds.iot.dianne.api.nn.learn.Learner;
import be.iminds.iot.dianne.api.nn.learn.Processor;
import be.iminds.iot.dianne.api.nn.module.Input;
import be.iminds.iot.dianne.api.nn.module.Module;
import be.iminds.iot.dianne.api.nn.module.Output;
import be.iminds.iot.dianne.api.nn.module.Trainable;
import be.iminds.iot.dianne.api.nn.module.Module.Mode;
import be.iminds.iot.dianne.api.nn.module.dto.ModuleInstanceDTO;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkDTO;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkInstanceDTO;
import be.iminds.iot.dianne.api.nn.platform.NeuralNetwork;
import be.iminds.iot.dianne.api.nn.platform.NeuralNetworkManager;
import be.iminds.iot.dianne.api.nn.runtime.ModuleManager;
import be.iminds.iot.dianne.api.repository.DianneRepository;
import be.iminds.iot.dianne.api.rl.ExperiencePool;
import be.iminds.iot.dianne.nn.learn.processors.AbstractProcessor;
import be.iminds.iot.dianne.nn.learn.processors.MomentumProcessor;
import be.iminds.iot.dianne.nn.learn.processors.RegularizationProcessor;
import be.iminds.iot.dianne.rl.learn.processors.TimeDifferenceProcessor;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;

@Component
public class DeepQLearner implements Learner {

	private DataLogger logger;
	private String[] logLabels = new String[]{"minibatch time (ms)"};
	
	private TensorFactory factory;
	private NeuralNetworkManager dianne;
	private DianneRepository repository;
	private Map<String, ExperiencePool> pools = new HashMap<String, ExperiencePool>();

	private NeuralNetwork nn;
	private NeuralNetwork target;

	private Map<UUID, Tensor> previousParameters;
	
	private Thread learningThread;
	private volatile boolean learning;
	
	private Processor processor;
	private ExperiencePool pool; 

	private String tag = "learn";
	private int targetInterval = 1000;
	private int syncInterval = 1;
	private int minSamples = 10000;
	private boolean clean = false;

	@Reference
	void setTensorFactory(TensorFactory factory) {
		this.factory = factory;
	}

	@Reference
	void setNeuralNetworkManager(NeuralNetworkManager nnm) {
		this.dianne = nnm;
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

		if (config.containsKey("targetInterval"))
			targetInterval = Integer.parseInt(config.get("targetInterval"));
		
		if (config.containsKey("syncInterval"))
			syncInterval = Integer.parseInt(config.get("syncInterval"));
		
		if (config.containsKey("minSamples"))
			minSamples = Integer.parseInt(config.get("minSamples"));
		
		if (config.containsKey("clean"))
			clean = Boolean.parseBoolean(config.get("clean"));
		
		System.out.println("Learner Configuration");
		System.out.println("=====================");
		System.out.println("* tag = "+tag);
		System.out.println("* targetInterval = "+targetInterval);
		System.out.println("* syncInterval = "+syncInterval);
		System.out.println("* minSamples = "+minSamples);
		System.out.println("* clean = "+clean);
		System.out.println("---");
		

		NeuralNetworkInstanceDTO nni = dianne.deployNeuralNetwork(nnName, "Deep Q Learning NN instance");
		nn = dianne.getNeuralNetwork(nni.id);
		nn.getInput().setMode(EnumSet.of(Mode.BLOCKING));

		
		NeuralNetworkInstanceDTO targeti = dianne.deployNeuralNetwork(nnName, "Deep Q Learning Target NN instance");
		target = dianne.getNeuralNetwork(targeti.id);
		target.getInput().setMode(EnumSet.of(Mode.BLOCKING));

		
		pool = pools.get(experiencePool);

		// create a Processor from config
		AbstractProcessor p = new TimeDifferenceProcessor(factory, nn,target, pool, config, logger);
		if(config.get("regularization")!=null){
			p = new RegularizationProcessor(p);
		}
		if(config.get("momentum")!=null){
			 p = new MomentumProcessor(p);
		}
		processor = p;

		learningThread = new Thread(new DeepQLearnerRunnable());
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

	private void loadParameters(NeuralNetwork... nns) {
		try {
			Map<UUID, Tensor> parameters = repository.loadParameters(nn.getTrainables().keySet(), tag);
			for(NeuralNetwork nn : nns){
				nn.setParameters(parameters);
			}
			previousParameters = parameters;
		} catch (Exception ex) {
			resetParameters();
		}
	}
	
	private void initializeParameters(){
		if(clean){
			// reset parameters
			resetParameters();
		} else {
			// load previous parameters
			loadParameters(nn, target);
		}
	}

	private void resetParameters(){
		nn.getTrainables().entrySet().stream().forEach(e -> e.getValue().reset());
		
		// collect parameters
		Map<UUID, Tensor> parameters = nn.getParameters();
		repository.storeParameters(parameters, tag);
		
		// copy to target
		target.setParameters(nn.getParameters());
		
		// copy to previousParameters
		previousParameters =  nn.getParameters().entrySet().stream().collect(Collectors.toMap(e -> e.getKey(), e -> e.getValue().copyInto(null)));
	}
	
	private void publishDeltaParameters() {
		Map<UUID, Tensor> deltaParameters = nn.getParameters().entrySet().stream()
				.collect(Collectors.toMap(e -> e.getKey(), e -> factory.getTensorMath().sub(null,
						e.getValue(), previousParameters.get(e.getKey()))));

		repository.accParameters(deltaParameters, tag);
		
	}

	private class DeepQLearnerRunnable implements Runnable {

		private static final double alpha = 1e-2;

		@Override
		public void run() {
			double error = 0, avgError = 0;
			long timestamp = System.currentTimeMillis();
			
			initializeParameters();
			
			// wait until pool has some samples
			if(pool.size() < minSamples){
				System.out.println("Experience pool has too few samples, waiting a bit to start learning...");
				while(pool.size() < minSamples){
					try {
						Thread.sleep(5000);
					} catch (InterruptedException e) {
					}
				}
			}
			System.out.println("Start learning...");
			
			for (long i = 1; learning; i++) {
				nn.getTrainables().values().stream().forEach(Trainable::zeroDeltaParameters);
				
				pool.lock();
				try {
					error = processor.processNext();
					if(Double.isInfinite(error) || Double.isNaN(error)){
						System.out.println(i+" ERROR IS "+error);
					}
					
				} finally {
					pool.unlock();
				}
				
				avgError = (1 - alpha) * avgError + alpha * error;

				if(logger!=null){
					long t = System.currentTimeMillis();
					logger.log("TIME", logLabels, (float)(t-timestamp));
					timestamp = t;
				}

				nn.getTrainables().values().stream().forEach(Trainable::updateParameters);

				if (targetInterval > 0 && i % targetInterval == 0) {
					publishDeltaParameters();
					loadParameters(nn, target);
				} else if(syncInterval > 0 && i % syncInterval == 0){
					publishDeltaParameters();
					loadParameters(nn);
				}
			}

			publishDeltaParameters();
		}

	}
	
	@Reference(cardinality = ReferenceCardinality.OPTIONAL)
	public void setDataLogger(DataLogger l){
		this.logger = l;
	}
}
