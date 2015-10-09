package be.iminds.iot.dianne.nn.learn;

import java.util.EnumSet;
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
import be.iminds.iot.dianne.api.nn.Dianne;
import be.iminds.iot.dianne.api.nn.NeuralNetwork;
import be.iminds.iot.dianne.api.nn.learn.LearnProgress;
import be.iminds.iot.dianne.api.nn.learn.Learner;
import be.iminds.iot.dianne.api.nn.learn.Processor;
import be.iminds.iot.dianne.api.nn.module.Module.Mode;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkInstanceDTO;
import be.iminds.iot.dianne.nn.learn.factory.LearnerFactory;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;

@Component
public class SGDLearner implements Learner {
	
	protected DataLogger logger;
	
	protected TensorFactory factory;
	protected Dianne dianne;
	protected Map<String, Dataset> datasets = new HashMap<String, Dataset>();
	
	protected Thread learnerThread = null;
	protected volatile boolean learning = false;
	protected Processor processor;
	
	protected int syncInterval = 1000;
	protected boolean clean = false;
	
	// the network we are currently training
	protected NeuralNetwork nn;
	
	protected String tag = "learn";
	
	// initial parameters
	protected Map<UUID, Tensor> parameters = null;
	
	private static final float alpha = 1e-2f;
	protected float error = 0;
	protected long i = 0;
	
	@Override
	public LearnProgress getProgress(){
		return new LearnProgress(i, error);
	}
	
	@Override
	public void learn(NeuralNetworkInstanceDTO nni, String dataset,
			Map<String, String> config) throws Exception {
		if(learning){
			throw new Exception("Already running a learning session here");
		}
		
		if(config.containsKey("tag")){
			tag = config.get("tag"); 
		}
		
		if(config.containsKey("syncInterval")){
			syncInterval = Integer.parseInt(config.get("syncInterval")); 
		}
		
		if (config.containsKey("clean")){
			clean = Boolean.parseBoolean(config.get("clean"));
		}
		
		System.out.println("Learner Configuration");
		System.out.println("=====================");
		System.out.println("* dataset = "+dataset);
		System.out.println("* tag = "+tag);
		System.out.println("* syncInterval = "+syncInterval);
		System.out.println("* clean = " +clean);
		System.out.println("---");
		
		// Fetch the dataset
		Dataset d = datasets.get(dataset);
		if(d==null){
			throw new Exception("Dataset "+dataset+" not available");
		}
		
		nn = dianne.getNeuralNetwork(nni);
		nn.getModules().values().stream().forEach(m -> m.setMode(EnumSet.of(Mode.BLOCKING)));
		
		// initialize nn parameters
		if(clean){
			nn.resetParameters();
		} else {
			loadParameters();
		}
		
		// first get parameters for preprocessing?
		nn.getPreprocessors().values().stream().forEach(p -> {
			if(!p.isPreprocessed())
				p.preprocess(d);
			}
		);
		
		// create a Processor from config
		processor = LearnerFactory.createProcessor(factory, nn, d, config, logger);
		
		learnerThread = new Thread(new Runnable() {
			
			@Override
			public void run() {
				learning = true;
				i = 0;
				float err = 0;

				do {
					err = processor.processNext();
					if(i==0){
						error = err;
					} else {
						error = (1 - alpha) * error + alpha * err;
					}

					System.out.println(err+" - "+error);
					
					if(error >= 0){
						
						nn.getTrainables().entrySet().stream().forEach(e -> {
							e.getValue().updateParameters(1.0f);
							e.getValue().zeroDeltaParameters();
						});
						
						if(syncInterval>0){
							if(i % syncInterval == 0){
								// publish weights
								publishParameters();
							}
						}
					}
					
					i++;
				} while(learning);
				System.out.println("Stopped learning");
				publishParameters();
				learning = false;
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
			parameters = nn.loadParameters(tag);
		} catch(Exception ex){
			// if no initial parameters available, publish the random initialize parameters of this instance as first parameters
			nn.storeParameters(tag);
			parameters =  nn.getParameters().entrySet().stream().collect(Collectors.toMap(e -> e.getKey(), e -> e.getValue().copyInto(null)));
			System.out.println("No initial parameters available, publish these random values as initial parameters...");
		}
	}
	
	protected void publishParameters(){
		System.out.println("Publish parameters");
		
		// collect all parameter deltas to update repository
		Map<UUID, Tensor> newParameters = nn.getParameters();
				
		if(parameters!=null){
			// publish delta
			nn.storeDeltaParameters(parameters, tag);
		} else {
			// just publish initial values
			nn.storeParameters(tag);
		}
		
		// fetch update again from repo (could be merged from other learners)
		loadParameters();
	}
	
	@Reference
	void setTensorFactory(TensorFactory f){
		this.factory = f;
	}
	
	@Reference
	void setDianne(Dianne d){
		dianne = d;
	}
	
	@Reference(cardinality=ReferenceCardinality.MULTIPLE, 
			policy=ReferencePolicy.DYNAMIC)
	void addDataset(Dataset dataset, Map<String, Object> properties){
		String name = (String) properties.get("name");
		this.datasets.put(name, dataset);
	}
	
	void removeDataset(Dataset dataset, Map<String, Object> properties){
		String name = (String) properties.get("name");
		this.datasets.remove(name);
	}
	
	@Reference(cardinality = ReferenceCardinality.OPTIONAL)
	void setDataLogger(DataLogger l){
		this.logger = l;
	}
}

