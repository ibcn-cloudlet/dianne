package be.iminds.iot.dianne.nn.eval;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.List;
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
import be.iminds.iot.dianne.api.nn.module.Module.Mode;
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
	
	protected String tag = null;
	
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
		
		
		if(config.containsKey("tag")){
			tag = config.get("tag"); 
		}
		
		System.out.println("Evaluator Configuration");
		System.out.println("=======================");
		System.out.println("* dataset = "+dataset);
		System.out.println("* tag = "+tag);
		
		
		int[] indices = null;
		String range = config.get("range");
		if(range!=null){
			indices = parseRange(range);
			
			System.out.println("Dataset range");
			System.out.println("* range = "+range);
			System.out.println("---");
		} else {
			int startIndex = 0;
			int endIndex = d.size();
			
			String start = config.get("startIndex");
			if(start!=null){
				startIndex = Integer.parseInt(start);
			}
			
			String end = config.get("endIndex");
			if(end!=null){
				endIndex = Integer.parseInt(end);
			}
			
			int index = startIndex;
			indices = new int[endIndex-startIndex];
			for(int i=0;i<indices.length;i++){
				indices[i] = index++;
			}
			
			System.out.println("Dataset range");
			System.out.println("* startIndex = "+startIndex);
			System.out.println("* endIndex = "+endIndex);
			System.out.println("---");
		}
		
		
		
		NeuralNetwork nn = null;
		try {
			nn = dianne.getNeuralNetwork(nni).getValue();
		} catch (Exception e) {
		}
		nn.getModules().values().stream().forEach(m -> m.setMode(EnumSet.of(Mode.BLOCKING)));
		
		if(tag==null){
			nn.loadParameters();
		} else {
			nn.loadParameters(tag);
		}
	
		Tensor confusion = null;
		List<Tensor> outputs = new ArrayList<Tensor>();
		long t1 = System.currentTimeMillis();
		for(int i=0;i<indices.length;i++){
			Tensor in = d.getInputSample(indices[i]);
			Tensor out = nn.forward(in);
			outputs.add(out);
			
			if(confusion==null){
				int outputSize = out.size();
				confusion = factory.createTensor(outputSize, outputSize);
				confusion.fill(0.0f);
			}
			
			int predicted = factory.getTensorMath().argmax(out);
			int real = factory.getTensorMath().argmax(d.getOutputSample(indices[i]));
				
			confusion.set(confusion.get(real, predicted)+1, real, predicted);
		}
		long t2 = System.currentTimeMillis();
		
		Evaluation e = new Evaluation(factory, confusion, outputs, t2-t1);
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
	
	private int[] parseRange(String range){
		ArrayList<Integer> list = new ArrayList<>();
		String[] subranges = range.split(",");
		for(String subrange : subranges){
			String[] s = subrange.split(":");
			if(s.length==2){
				for(int i=Integer.parseInt(s[0]);i<Integer.parseInt(s[1]);i++){
					list.add(i);
				}
			} else {
				list.add(Integer.parseInt(s[0]));
			}
		}
		int[] array = new int[list.size()];
		for(int i=0;i<list.size();i++){
			array[i] = list.get(i);
		}
		return array;
	}

}

