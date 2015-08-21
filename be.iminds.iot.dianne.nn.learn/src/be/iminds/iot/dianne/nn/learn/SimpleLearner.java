package be.iminds.iot.dianne.nn.learn;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;

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
import be.iminds.iot.dianne.nn.learn.processors.MomentumProcessor;
import be.iminds.iot.dianne.nn.learn.processors.RandomBatchProcessor;
import be.iminds.iot.dianne.nn.learn.processors.RegularizationProcessor;
import be.iminds.iot.dianne.tensor.TensorFactory;

@Component
public class SimpleLearner implements Learner {

	private TensorFactory factory;
	private DianneRepository repository;
	private ModuleManager runtime;
	private Map<String, Dataset> datasets = new HashMap<String, Dataset>();
	
	private Thread learnerThread = null;
	private volatile boolean learning = false;
	
	private int updateInterval = 0;
	
	@Override
	public void learn(String nnName, String dataset,
			Map<String, String> config) throws Exception {
		if(learning){
			throw new Exception("Already running a learning session here");
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
		NeuralNetworkInstanceDTO nni = new NeuralNetworkInstanceDTO(nnId, nnName, moduleInstances);

		Input input = null;
		Output output = null;
		Map<UUID, Trainable> toTrain = new HashMap<>();
		Set<Preprocessor> preprocessing = new HashSet<>();
		
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
		
		// first get parameters for preprocessing?
		preprocessing.stream().forEach(p -> {
			if(!p.isPreprocessed())
				p.preprocess(d);
			}
		);
		
		// create a Processor from config
		// for now just fixed
		Processor p = new MomentumProcessor(new RegularizationProcessor(new RandomBatchProcessor(factory, input, output, toTrain, d, config)));
		
		
		learnerThread = new Thread(new Runnable() {
			
			@Override
			public void run() {
				learning = true;
				int i = 0;
				float runningAvg = 0;
				float error = 0;

				do {
					i++;
					error = p.processNext();
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

	private void publishParameters(){
		System.out.println("Publish parameters");
		
		// TODO collect all parameter deltas to update repository
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

