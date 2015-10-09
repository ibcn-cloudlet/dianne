package be.iminds.iot.dianne.nn.eval;

import java.util.HashMap;
import java.util.Map;
import java.util.UUID;

import org.osgi.framework.BundleContext;
import org.osgi.framework.Constants;
import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;
import org.osgi.service.component.annotations.ReferenceCardinality;
import org.osgi.service.component.annotations.ReferencePolicy;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.log.DataLogger;
import be.iminds.iot.dianne.api.nn.Dianne;
import be.iminds.iot.dianne.api.nn.NeuralNetwork;
import be.iminds.iot.dianne.api.nn.eval.Evaluation;
import be.iminds.iot.dianne.api.nn.eval.Evaluator;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkInstanceDTO;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;

@Component(property={"aiolos.unique=true"})
public class ArgMaxEvaluator implements Evaluator {
	
	protected UUID evaluatorId;
	
	protected DataLogger logger;
	
	protected TensorFactory factory;
	protected Dianne dianne;
	protected Map<String, Dataset> datasets = new HashMap<String, Dataset>();
	
	@Override
	public UUID getEvaluatorId(){
		return evaluatorId;
	}

	@Override
	public synchronized Evaluation eval(NeuralNetworkInstanceDTO nni, String dataset,
			Map<String, String> config) throws Exception {
		
		// Fetch the dataset
		Dataset d = datasets.get(dataset);
		if(d==null){
			throw new Exception("Dataset "+dataset+" not available");
		}
		
		int startIndex = 0;
		int endIndex = d.size();
		
		String tag = null;
		if(config.containsKey("tag")){
			tag = config.get("tag"); 
		}
		
		if(config.containsKey("startIndex")){
			startIndex = Integer.parseInt(config.get("startIndex")); 
		}
		
		if(config.containsKey("endIndex")){
			endIndex = Integer.parseInt(config.get("endIndex")); 
		}
		
		System.out.println("Evaluator Configuration");
		System.out.println("=======================");
		System.out.println("* dataset = "+dataset);
		System.out.println("* tag = "+tag);
		System.out.println("* startIndex = "+startIndex);
		System.out.println("* endIndex = " +endIndex);
		System.out.println("---");
		
		
		NeuralNetwork nn = dianne.getNeuralNetwork(nni);
		nn.loadParameters(tag);
	
		Tensor confusion = null;
		long t1 = System.currentTimeMillis();
		for(int i=startIndex;i<endIndex;i++){
			Tensor in = d.getInputSample(i);
			Tensor out = nn.forward(in);
			
			if(confusion==null){
				int outputSize = out.size();
				confusion = factory.createTensor(outputSize, outputSize);
				confusion.fill(0.0f);
			}
			
			int predicted = factory.getTensorMath().argmax(out);
			int real = factory.getTensorMath().argmax(d.getOutputSample(i));
				
			confusion.set(confusion.get(real, predicted)+1, real, predicted);
		}
		long t2 = System.currentTimeMillis();
		
		Evaluation e = new Evaluation(factory, confusion, t2-t1);
		return e;
	}

	@Activate
	public void activate(BundleContext context){
		this.evaluatorId = UUID.fromString(context.getProperty(Constants.FRAMEWORK_UUID));
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

