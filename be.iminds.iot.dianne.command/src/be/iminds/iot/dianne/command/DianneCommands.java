package be.iminds.iot.dianne.command;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Dictionary;
import java.util.HashMap;
import java.util.Hashtable;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.UUID;

import org.osgi.framework.BundleContext;
import org.osgi.framework.InvalidSyntaxException;
import org.osgi.framework.ServiceReference;
import org.osgi.framework.ServiceRegistration;
import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;
import org.osgi.service.component.annotations.ReferenceCardinality;
import org.osgi.service.component.annotations.ReferencePolicy;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.nn.module.ForwardListener;
import be.iminds.iot.dianne.api.nn.module.Input;
import be.iminds.iot.dianne.api.nn.module.Module;
import be.iminds.iot.dianne.api.nn.module.Output;
import be.iminds.iot.dianne.api.nn.module.dto.ModuleInstanceDTO;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkInstanceDTO;
import be.iminds.iot.dianne.api.nn.platform.NeuralNetworkManager;
import be.iminds.iot.dianne.api.nn.runtime.ModuleManager;
import be.iminds.iot.dianne.api.repository.DianneRepository;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;

@Component(
		service=Object.class,
		property={"osgi.command.scope=dianne",
				  "osgi.command.function=datasets",
				  "osgi.command.function=runtimes",
				  "osgi.command.function=nn",
				  "osgi.command.function=nnAvailable",
				  "osgi.command.function=nnDeploy",
				  "osgi.command.function=nnUndeploy",
				  "osgi.command.function=sample",
				  "osgi.command.function=eval"},
		immediate=true)
public class DianneCommands {

	private static Random rand = new Random(System.currentTimeMillis());
	
	BundleContext context;
	
	// Dianne components
	TensorFactory factory; 
	
	Map<String, Dataset> datasets = Collections.synchronizedMap(new HashMap<String, Dataset>());
	List<ModuleManager> runtimes = Collections.synchronizedList(new ArrayList<ModuleManager>());
	DianneRepository repository;
	NeuralNetworkManager dianne;
	
	// State
	List<NeuralNetworkInstanceDTO> nns = new ArrayList<NeuralNetworkInstanceDTO>();
	Map<String, NeuralNetworkInstanceDTO> map = new HashMap<String, NeuralNetworkInstanceDTO>();
	
	// Separate aggregation for training commands
	private DianneTrainCommands training = null;
	
	@Activate
	public void activate(BundleContext context){
		this.context = context;
		
		try {
			training = new DianneTrainCommands(this);
		} catch(NoClassDefFoundError e){
			//ignore
		}
	}
	
	public void datasets(){
		if(datasets.size()==0){
			System.out.println("No datasets available");
			return;
		}
		
		System.out.println("Available datasets:");
		synchronized(datasets){
			int i = 0;
			for(Dataset dataset : datasets.values()){
				System.out.println("["+(i++)+"] "+dataset.getName()+"\t"+dataset.size()+" samples");
			}
		}
	}
	
	public void runtimes(){
		if(runtimes.size()==0){
			System.out.println("No runtimes available");
			return;
		}
		
		System.out.println("Available Dianne runtimes:");
		synchronized(runtimes){
			int i = 0;
			for(ModuleManager runtime : runtimes){
				System.out.println("["+(i++)+"] "+runtime.getRuntimeId());
			}
		}
	}
	
	public void nnAvailable(){
		List<String> nns = repository.availableNeuralNetworks();
		if(nns.size()==0){
			System.out.println("No neural networks available");
			return;
		}
		
		System.out.println("Available neural networks:");
		int i=0;
		for(String nn : nns){
			System.out.println("["+(i++)+"] "+nn);
		}
	}
	
	public void nn(){
		if(nns.size()==0){
			System.out.println("No neural networks deployed");
			return;
		}
		
		System.out.println("Deployed neural networks:");
		int i=0;
		for(NeuralNetworkInstanceDTO nn : nns){
			System.out.println("["+(i++)+"] "+nn.id+"\t"+nn.name);
		}
		
	}
	
	public void nn(int index){
		if(index >= nns.size()){
			System.out.println("No neural network deployed with index "+index);
			return;
		}
		NeuralNetworkInstanceDTO nn = nns.get(index);
		printNN(nn);
	}
	
	public void nn(String id){
		NeuralNetworkInstanceDTO nn = map.get(id);
		if(nn==null){
			System.out.println("No neural network deployed with id "+id);
			return;
		}
		printNN(nn);
	}
	
	private void printNN(NeuralNetworkInstanceDTO nn){
		System.out.println(nn.id.toString()+" ("+nn.name+")");
		for(ModuleInstanceDTO m: nn.modules){
			System.out.println("* "+m.moduleId+" deployed at "+m.runtimeId);
		}
	}
	
	public void nnDeploy(String name){
		deploy(name, runtimes.get(0).getRuntimeId());
	}
	
	public void nnDeploy(String name, String id){
		deploy(name, UUID.fromString(id));
	}
	
	public void nnDeploy(String name, int index){
		deploy(name, runtimes.get(index).getRuntimeId());
	}
	
	private synchronized void deploy(String name, UUID runtimeId){
		try {
			NeuralNetworkInstanceDTO nn = dianne.deployNeuralNetwork(name, runtimeId);
			nns.add(nn);
			map.put(nn.id.toString(), nn);
			System.out.println("Deployed instance of "+nn.name+" ("+nn.id.toString()+")");
		} catch (InstantiationException e) {
			System.out.println("Error deploying instance of "+name);
			e.printStackTrace();
		}	
	}
	
	public void nnUndeploy(String nnId){
		NeuralNetworkInstanceDTO nn = map.get(nnId);
		if(nn==null){
			System.out.println("No neural network deployed with id "+nnId);
			return;
		}
		undeploy(nn);
	}
	
	public void nnUndeploy(int index){
		if(index >= nns.size()){
			System.out.println("No neural network with index "+index);
			return;
		}
		NeuralNetworkInstanceDTO nn = nns.get(index);
		undeploy(nn);
	}
	
	private void undeploy(NeuralNetworkInstanceDTO nn){
		dianne.undeployNeuralNetwork(nn);
		nns.remove(nn);
		map.remove(nn.id);
	}
	
	public void sample(String dataset, String nnId, int sample, String...tags){
		Dataset d = datasets.get(dataset);
		if(d==null){
			System.out.println("Dataset "+dataset+" not available");
			return;
		}
		
		final int index = sample == -1 ? rand.nextInt(d.size()) : sample;
		
		UUID inputId = getInputId(nnId);
		if(inputId==null){
			System.out.println("No Input module found for neural network "+nnId);
			return;
		}
		
		UUID outputId = getOutputId(nnId);
		if(outputId==null){
			System.out.println("No Output module found for neural network "+nnId);
			return;
		}
		
		ServiceReference refOutput = getModule(UUID.fromString(nnId), outputId);
		if(refOutput==null){
			System.out.println("Output module "+outputId+" not found");
			return;
		}
		
		Output output = (Output) context.getService(refOutput);
		final String[] labels = output.getOutputLabels();
		context.ungetService(refOutput);
		
		
		ServiceReference refInput = getModule(UUID.fromString(nnId), inputId);
		if(refInput==null){
			System.out.println("Input module "+inputId+" not found");
			return;
		}
		
		// register outputlistener
		Dictionary<String, Object> properties = new Hashtable<String, Object>();
		properties.put("targets", new String[]{nnId.toString()+":"+outputId.toString()});
		final ForwardListener printer = new ForwardListener() {
			@Override
			public void onForward(Tensor output, String... tags) {
				int clazz = factory.getTensorMath().argmax(output);
				float max = factory.getTensorMath().max(output);
				String label = labels[clazz];
				
				System.out.println("Sample "+index+" (with tags "+Arrays.toString(tags)+") classified as: "+label+" (probability: "+max+")");
				
				synchronized(DianneCommands.this){
					DianneCommands.this.notifyAll();
				}
			}
		};
		ServiceRegistration reg = context.registerService(ForwardListener.class, printer, properties);
		
		// get input and forward
		Input input = (Input) context.getService(refInput);
		
		try {
			Tensor t = d.getInputSample(index);
			long t1 = System.currentTimeMillis();
			input.input(t, tags);
			synchronized(this){
				try {
					this.wait();
				} catch (InterruptedException e) {
				}
			}
			long t2 = System.currentTimeMillis();
			System.out.println("Forward time: "+(t2-t1)+" ms");
		} catch(Throwable t){
			t.printStackTrace();
		} finally {
			// cleanup
			context.ungetService(refInput);
			reg.unregister();
		}
		
	}
	
	public void sample(String dataset, String nnId, String...tags){
		sample(dataset, nnId, -1, tags);
	}
	
	
	public void eval(String dataset, String nnId, int start, int end){
		if(training == null){
			System.out.println("Training/Evaluation functions unavailable");
			return;
		}

		training.eval(dataset, nnId, start, end);
	}
	
	UUID getInputId(String nnId){
		NeuralNetworkInstanceDTO nn = map.get(nnId);
		if(nn==null)
			return null;
		
		for(ModuleInstanceDTO m : nn.modules){
			if(m.module.type.equals("Input")){
				return m.moduleId;
			}
		}
		
		return null;
	}
	
	UUID getOutputId(String nnId){
		NeuralNetworkInstanceDTO nn = map.get(nnId);
		if(nn==null)
			return null;
		
		for(ModuleInstanceDTO m : nn.modules){
			if(m.module.type.equals("Output")){
				return m.moduleId;
			}
		}
		
		return null;
	}
	
	ServiceReference getModule(UUID nnId, UUID moduleId){
		try {
			ServiceReference[] refs = context.getAllServiceReferences(Module.class.getName(), "(&(module.id="+moduleId.toString()+")(nn.id="+nnId.toString()+"))");
			if(refs!=null){
				return refs[0];
			}
		} catch (InvalidSyntaxException e) {
			e.printStackTrace();
		}
		return null;
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
	
	@Reference(cardinality=ReferenceCardinality.MULTIPLE, 
			policy=ReferencePolicy.DYNAMIC)
	public void addModuleManager(ModuleManager runtime, Map<String, Object> properties){
		this.runtimes.add(runtime);
	}
	
	public void removeModuleManager(ModuleManager runtime, Map<String, Object> properties){
		this.runtimes.remove(runtime);
	}
	
	@Reference
	public void setDianneRepository(DianneRepository repo){
		this.repository = repo;
	}
	
	@Reference
	public void setNeuralNetworkManager(NeuralNetworkManager nnMgr){
		this.dianne = nnMgr;
	}

	@Reference
	public void setTensorFactory(TensorFactory f){
		this.factory = f;
	}
}
