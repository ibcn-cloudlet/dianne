package be.iminds.iot.dianne.nn.learn;

import java.util.HashMap;
import java.util.Map;
import java.util.UUID;
import java.util.stream.Collectors;

import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;
import org.osgi.service.component.annotations.ReferenceCardinality;
import org.osgi.service.component.annotations.ReferencePolicy;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.log.DataLogger;
import be.iminds.iot.dianne.api.nn.learn.Learner;
import be.iminds.iot.dianne.api.nn.learn.Processor;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkInstanceDTO;
import be.iminds.iot.dianne.api.nn.platform.NeuralNetwork;
import be.iminds.iot.dianne.api.nn.platform.Dianne;
import be.iminds.iot.dianne.api.repository.DianneRepository;
import be.iminds.iot.dianne.nn.learn.processors.AbstractProcessor;
import be.iminds.iot.dianne.nn.learn.processors.MomentumProcessor;
import be.iminds.iot.dianne.nn.learn.processors.RegularizationProcessor;
import be.iminds.iot.dianne.nn.learn.processors.StochasticGradientDescentProcessor;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;

@Component
public class SGDLearner implements Learner {
	
	protected DataLogger logger;
	
	protected TensorFactory factory;
	protected DianneRepository repository;
	protected Dianne dianne;
	protected Map<String, Dataset> datasets = new HashMap<String, Dataset>();
	
	protected Thread learnerThread = null;
	protected volatile boolean learning = false;
	protected Processor processor;
	
	protected int updateInterval = 1000;
	
	// the network we are currently training
	protected NeuralNetwork nn;
	
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
		NeuralNetworkInstanceDTO nni = dianne.deployNeuralNetwork(nnName, "Learning NN instance");
		nn = dianne.getNeuralNetwork(nni.id);
		
		// load parameters
		loadParameters();
		
		// first get parameters for preprocessing?
		nn.getPreprocessors().values().stream().forEach(p -> {
			if(!p.isPreprocessed())
				p.preprocess(d);
			}
		);
		
		// create a Processor from config
		AbstractProcessor p = new StochasticGradientDescentProcessor(factory, nn, d, config, logger);
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
						
						nn.getTrainables().entrySet().stream().forEach(e -> {
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
			parameters = repository.loadParameters(nn.getTrainables().keySet(), tag);
			nn.setParameters(parameters);
		} catch(Exception e){
			// if no initial parameters available, publish the random initialize parameters of this instance as first parameters?
			publishParameters();
			System.out.println("No initial parameters available, publish these random values as initial parameters...");
		}
	}
	
	protected void publishParameters(){
		System.out.println("Publish parameters");
		
		// collect all parameter deltas to update repository
		Map<UUID, Tensor> newParameters = nn.getParameters();
				
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
	void setDianne(Dianne d){
		dianne = d;
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
	
	@Reference(cardinality = ReferenceCardinality.OPTIONAL)
	public void setDataLogger(DataLogger l){
		this.logger = l;
	}
}

