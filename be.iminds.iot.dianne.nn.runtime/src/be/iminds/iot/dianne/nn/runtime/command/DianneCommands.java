package be.iminds.iot.dianne.nn.runtime.command;

import java.util.Arrays;
import java.util.Collections;
import java.util.Dictionary;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.UUID;

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
import be.iminds.iot.dianne.api.nn.runtime.ModuleManager;
import be.iminds.iot.dianne.api.repository.DianneRepository;
import be.iminds.iot.dianne.nn.runtime.util.DianneJSONParser;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;

@Component(
		service=Object.class,
		property={"osgi.command.scope=dianne",
				  "osgi.command.function=datasets",
				  "osgi.command.function=networks",
				  "osgi.command.function=modules",
				  "osgi.command.function=devices",
				  "osgi.command.function=loadNN",
				  "osgi.command.function=unloadNN",
				  "osgi.command.function=loadDataset",
				  "osgi.command.function=unloadDataset",
				  "osgi.command.function=sample",
				  "osgi.command.function=eval"},
		immediate=true)
public class DianneCommands {
	private static Random rand = new Random(System.currentTimeMillis());
	
	TensorFactory factory; 
	
	// for now only deploy one network at a time
	String network = null;
	// for now only load one dataset at a time
	Dataset dataset = null;
	
	Map<String, Dataset> datasets = Collections.synchronizedMap(new HashMap<String, Dataset>());
	DianneRepository repository;
	// for now only one runtime is used
	Map<String, ModuleManager> runtimes = Collections.synchronizedMap(new HashMap<String, ModuleManager>());
	Map<String, Module> modules = Collections.synchronizedMap(new HashMap<String, Module>());

	// for now only support single Input/Output 
	Input input = null;
	Output output = null;
	
	private DianneTrainCommands training = null;
	
	@Activate
	public void activate(){
		try {
			training = new DianneTrainCommands(this);
		} catch(NoClassDefFoundError e){
			//ignore
		}
	}
	
	public void datasets(){
		System.out.println("Available datasets:");
		synchronized(datasets){
			for(Dataset dataset : datasets.values()){
				System.out.println("* "+dataset.getName()+"\t"+dataset.size()+" samples");
			}
		}
	}
	
	public void networks(){
		System.out.println("Available neural networks:");
		for(String network : repository.networks()){
			System.out.println("* "+network);
		}
	}
	
	public void modules(){
		System.out.println("Deployed modules:");
		synchronized(modules){
			Iterator<Entry<String, Module>> it = modules.entrySet().iterator();
			while(it.hasNext()){
				Entry<String, Module> entry = it.next();
				System.out.println("* "+entry.getKey()+"\t"+entry.getValue().getClass().getName());
			}
		}
	}
	
	public void devices(){
		System.out.println("Available devices:");
		synchronized(runtimes){
			for(String device : runtimes.keySet()){
				System.out.println("* "+device);
			}
		}
	}
	
	public synchronized void loadNN(String network){
		if(this.network==null){
			this.network=network;
		} else {
			System.out.println("Already deployed a network: "+network);
			return;
		}
		try {
			String json = repository.loadNetwork(network);
			List<Dictionary<String, Object>> modules = DianneJSONParser.parseJSON(json);
			
			for(Dictionary<String, Object> module : modules){
				ModuleManager m = runtimes.get("localhost");
				m.deployModule(module);
			}
			
		} catch(Exception e){
			System.out.println("Failed to load network "+network);
		}
	}
	
	public synchronized void unload(){
		synchronized(runtimes){
			for(ModuleManager m : runtimes.values()){
				for(UUID id : m.getModules()){
					m.undeployModule(id);
				}
			}
		}
		this.network = null;
	}
	
	public synchronized void loadDataset(String dataset){
		if(this.dataset==null){
			this.dataset=datasets.get(dataset);
			if(this.dataset==null){
				System.out.println("Dataset "+dataset+" not found");
			} else {
				System.out.println("Succesfully loaded dataset "+dataset);
			}
		} else {
			System.out.println("Already deployed a dataset ("+this.dataset.getName()+") ... unload first with unloadDataset");
		}
	}
	
	public synchronized void unloadDataset(){
		this.dataset = null;
	}
	
	public void sample(String... tags){
		if(dataset==null){
			System.out.println("No dataset loaded, load one first with loadDataset");
			return;
		}
		int index = rand.nextInt(dataset.size());
		sample(index, tags);
	}
	
	public void sample(){
		sample(null);
	}
	
	public void sample(final int index, final String... tags){
		if(network==null){
			System.out.println("No neural network loaded, load one first with loadNN");
			return;
		} 
		if(dataset==null){
			System.out.println("No dataset loaded, load one first with loadDataset");
			return;
		}
		if(input==null){
			System.out.println("Loaded neural network has no valid Input module");
			return;
		}
		
		// TODO only works if output is deployed on this node, ok for now
		final ForwardListener printer = new ForwardListener() {
			
			@Override
			public void onForward(Tensor output, String... tags) {
				int clazz = factory.getTensorMath().argmax(output);
				float max = factory.getTensorMath().max(output);
				String label = dataset.getLabels()[clazz];
				System.out.println("Sample "+index+" (with tags "+Arrays.toString(tags)+") classified as: "+label+" (probability: "+max+")");
				
				synchronized(DianneCommands.this.output){
					DianneCommands.this.output.notifyAll();
				}
			}
		};
		
		output.addForwardListener(printer);
		Tensor t = dataset.getInputSample(index);
		long t1 = System.currentTimeMillis();
		input.input(t, tags);
		synchronized(output){
			try {
				output.wait();
			} catch (InterruptedException e) {
			}
		}
		long t2 = System.currentTimeMillis();
		output.removeForwardListener(printer);
		System.out.println("Forward time: "+(t2-t1)+" ms");
	}
	
	public void eval(int start, int end){
		if(training == null){
			System.out.println("Training/Evaluation functions unavailable");
			return;
		}

		training.eval(start, end);
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
	
	@Reference
	public void setDianneRepository(DianneRepository repo){
		this.repository = repo;
	}
	
	@Reference(cardinality=ReferenceCardinality.AT_LEAST_ONE, 
			policy=ReferencePolicy.DYNAMIC)
	public void addModuleManager(ModuleManager m, Map<String, Object> properties){
		String uuid = (String) properties.get("aiolos.framework.uuid");
		// for now only support localhost
		if(uuid==null){
			runtimes.put("localhost", m);
		} else {
			runtimes.put(uuid, m);
		}
	}
	
	public void removeModuleManager(ModuleManager m, Map<String, Object> properties){
		String uuid = (String) properties.get("aiolos.framework.uuid"); 
		if(uuid==null){
			runtimes.remove("localhost");
		} else {
			runtimes.remove(uuid);
		}
	}
	
	@Reference(
			cardinality=ReferenceCardinality.MULTIPLE, 
			policy=ReferencePolicy.DYNAMIC)
	public synchronized void addModule(Module module, Map<String, Object> properties){
		String id = (String)properties.get("module.id");
		modules.put(id, module);
		
		if(module instanceof Input){
			this.input = (Input)module;
		} else if(module instanceof Output){
			this.output = (Output)module;
		} 
	}

	public synchronized void removeModule(Module module, Map<String, Object> properties){
		String id = (String)properties.get("module.id");
		modules.remove(id);
		
		if(module instanceof Input){
			if(input==module){
				input = null;
			}
		} else if(module instanceof Output){
			if(output==module){
				output = null;
			}
		} 
	}
	
	@Reference
	public void setTensorFactory(TensorFactory f){
		this.factory = f;
	}
}
