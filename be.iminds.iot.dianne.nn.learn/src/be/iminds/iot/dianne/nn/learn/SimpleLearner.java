package be.iminds.iot.dianne.nn.learn;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import java.util.stream.Collectors;

import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;
import org.osgi.service.component.annotations.ReferenceCardinality;
import org.osgi.service.component.annotations.ReferencePolicy;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.nn.learn.Learner;
import be.iminds.iot.dianne.api.nn.learn.Processor;
import be.iminds.iot.dianne.api.nn.module.Input;
import be.iminds.iot.dianne.api.nn.module.Module;
import be.iminds.iot.dianne.api.nn.module.Output;
import be.iminds.iot.dianne.api.nn.module.Preprocessor;
import be.iminds.iot.dianne.api.nn.module.Trainable;
import be.iminds.iot.dianne.api.nn.module.dto.ModuleDTO;
import be.iminds.iot.dianne.api.nn.module.dto.ModuleInstanceDTO;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkDTO;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkInstanceDTO;
import be.iminds.iot.dianne.api.nn.runtime.ModuleManager;
import be.iminds.iot.dianne.api.repository.DianneRepository;
import be.iminds.iot.dianne.nn.learn.processors.AbstractProcessor;
import be.iminds.iot.dianne.nn.learn.processors.MomentumProcessor;
import be.iminds.iot.dianne.nn.learn.processors.RegularizationProcessor;
import be.iminds.iot.dianne.nn.learn.processors.StochasticGradientDescentProcessor;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;

@Component
public class SimpleLearner implements Learner {
	
	protected TensorFactory factory;
	protected DianneRepository repository;
	protected ModuleManager runtime;
	protected Map<String, Dataset> datasets = new HashMap<String, Dataset>();
	
	protected Thread learnerThread = null;
	protected volatile boolean learning = false;
	protected Processor processor;
	
	protected int updateInterval = 1000;
	
	// the network we are currently training
	protected NeuralNetworkInstanceDTO nni;
	protected Input input;
	protected Output output;
	protected Map<UUID, Trainable> toTrain;
	protected Set<Preprocessor> preprocessing;
	
	protected String tag = "learn";
	
	// initial parameters
	protected Map<UUID, Tensor> parameters = null;
	
	@Override
	public void learn(String nnName, String dataset,
			Map<String, String> config) throws Exception {
		if(learning){
			throw new Exception("Already running a learning session here");
		}
		
		if(config.containsKey("tag")){
			tag = config.get("tag"); 
		}
		
		// Fetch the dataset
		Dataset d = datasets.get(dataset);
		if(d==null){
			throw new Exception("Dataset "+dataset+" not available");
		}
		
		// Deploy an instance of nnName to train on
		NeuralNetworkDTO nn = repository.loadNeuralNetwork(nnName);
		UUID nnId = UUID.randomUUID();
		List<ModuleInstanceDTO> moduleInstances = new ArrayList<ModuleInstanceDTO>();
		for(ModuleDTO module : nn.modules){
			ModuleInstanceDTO instance = runtime.deployModule(module, nnId);
			moduleInstances.add(instance);
		}
		nni = new NeuralNetworkInstanceDTO(nnId, nnName, moduleInstances);

		input = null;
		output = null;
		toTrain = new HashMap<>();
		preprocessing = new HashSet<>();
		
		for(ModuleInstanceDTO mi : nni.modules){
			Module m = runtime.getModule(mi.moduleId, mi.nnId);
			if(m instanceof Input){
				input = (Input) m;
			} else if(m instanceof Output){
				output = (Output) m;
			} else if(m instanceof Trainable){
				toTrain.put(mi.moduleId, (Trainable)m);
			} else if(m instanceof Preprocessor){
				preprocessing.add((Preprocessor) m);
			}	
		}
		
		// load parameters
		loadParameters();
		
		
		// first get parameters for preprocessing?
		preprocessing.stream().forEach(p -> {
			if(!p.isPreprocessed())
				p.preprocess(d);
			}
		);
		
		// create a Processor from config
		AbstractProcessor p = new StochasticGradientDescentProcessor(factory, input, output, toTrain, d, config);
		if(config.get("regularization")!=null){
			p = new RegularizationProcessor(p);
		}
		if(config.get("momentum")!=null){
			 p = new MomentumProcessor(new RegularizationProcessor(p));
		}
		processor = p;
		
		learnerThread = new Thread(new Runnable() {
			
			@Override
			public void run() {
				learning = true;
				int i = 0;
				float runningAvg = 0;
				float error = 0;

				do {
					i++;
					error = processor.processNext();
					runningAvg+= error;
					
					System.out.println(error+" - "+runningAvg/i);
					
					if(error >= 0){
						
						toTrain.entrySet().stream().forEach(e -> {
							e.getValue().updateParameters(1.0f);
							e.getValue().zeroDeltaParameters();
						});
						
						if(updateInterval>0){
							if(i % updateInterval == 0){
								// publish weights
								publishParameters();
							}
						}
					}
					

				} while(learning && error >= 0);
				System.out.println("Stopped learning");
				publishParameters();
			}
		});
		learnerThread.start();
	}
	
	@Override
	public void stop() {
		if(learning){
			learning = false;
			learnerThread.interrupt();
			learnerThread = null;
		}
	}

	protected void loadParameters(){
		try {
			parameters = repository.loadParameters(toTrain.keySet(), tag);
			parameters.entrySet().stream().forEach( e -> toTrain.get(e.getKey()).setParameters(e.getValue()));
		} catch(Exception e){
			// if no initial parameters available, publish the random initialize parameters of this instance as first parameters?
			publishParameters();
			System.out.println("No initial parameters available, publish these random values as initial parameters...");
		}
	}
	
	protected void publishParameters(){
		System.out.println("Publish parameters");
		
		// collect all parameter deltas to update repository
		Map<UUID, Tensor> newParameters = toTrain.entrySet().stream().collect(
				Collectors.toMap(e -> e.getKey(), e -> e.getValue().getParameters()));
		
		if(parameters!=null){
			// compose diff
			Map<UUID, Tensor> accParameters = newParameters.entrySet().stream().collect(
					Collectors.toMap(e -> e.getKey(), 
									 e -> factory.getTensorMath().sub(null, newParameters.get(e.getKey()), parameters.get(e.getKey()))));
			repository.accParameters(accParameters, tag);
		} else {
			// just publish initial values
			repository.storeParameters(newParameters, tag);
		}
		
		// fetch update again from repo (could be merged from other learners)
		loadParameters();
	}
	
	@Reference
	void setTensorFactory(TensorFactory f){
		this.factory = f;
	}
	
	@Reference
	void setDianneRepository(DianneRepository r){
		this.repository = r;
	}
	
	@Reference
	void setModuleManager(ModuleManager m){
		this.runtime = m;
	}
	
	@Reference(cardinality=ReferenceCardinality.MULTIPLE, 
			policy=ReferencePolicy.DYNAMIC)
	public void addDataset(Dataset dataset, Map<String, Object> properties){
		String name = (String) properties.get("name");
		this.datasets.put(name, dataset);
	}
	
	public void removeDataset(Dataset dataset, Map<String, Object> properties){
		String name = (String) properties.get("name");
		this.datasets.remove(name);
	}
}

