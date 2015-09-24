package be.iminds.iot.dianne.command;

import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Dictionary;
import java.util.HashMap;
import java.util.Hashtable;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.UUID;

import org.osgi.framework.BundleContext;
import org.osgi.framework.ServiceRegistration;
import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;
import org.osgi.service.component.annotations.ReferenceCardinality;
import org.osgi.service.component.annotations.ReferencePolicy;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.nn.module.dto.ModuleInstanceDTO;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkInstanceDTO;
import be.iminds.iot.dianne.api.nn.platform.Dianne;
import be.iminds.iot.dianne.api.nn.platform.NeuralNetwork;
import be.iminds.iot.dianne.api.repository.DianneRepository;
import be.iminds.iot.dianne.api.repository.RepositoryListener;
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
	DianneRepository repository;
	Dianne dianne;
	
	// State
	Map<UUID, ServiceRegistration> repoListeners = new HashMap<UUID, ServiceRegistration>();
	
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
		if(dianne.getRuntimes().size()==0){
			System.out.println("No runtimes available");
			return;
		}
		
		System.out.println("Available Dianne runtimes:");
		int i = 0;
		for(UUID runtime : dianne.getRuntimes()){
			System.out.println("["+(i++)+"] "+runtime);
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
		List<NeuralNetworkInstanceDTO> nns = dianne.getNeuralNetworkInstances();
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
		List<NeuralNetworkInstanceDTO> nns = dianne.getNeuralNetworkInstances();
		if(index >= nns.size()){
			System.out.println("No neural network deployed with index "+index);
			return;
		}
		NeuralNetworkInstanceDTO nn = nns.get(index);
		printNN(nn);
	}
	
	public void nn(String id){
		NeuralNetworkInstanceDTO nn = dianne.getNeuralNetworkInstance(UUID.fromString(id));
		if(nn==null){
			System.out.println("No neural network deployed with id "+id);
			return;
		}
		printNN(nn);
	}
	
	private void printNN(NeuralNetworkInstanceDTO nn){
		System.out.println(nn.id.toString()+" ("+nn.name+")");
		for(ModuleInstanceDTO m: nn.modules.values()){
			System.out.println("* "+m.moduleId+" deployed at "+m.runtimeId);
		}
	}
	
	public void nnDeploy(String name){
		deploy(name, dianne.getRuntimes().get(0));
	}
	
	public void nnDeploy(String name, String id){
		deploy(name, UUID.fromString(id));
	}
	
	public void nnDeploy(String name, int index){
		deploy(name, dianne.getRuntimes().get(index));
	}
	
	public void nnDeploy(String name, String id, String tag){
		NeuralNetworkInstanceDTO nn = deploy(name, UUID.fromString(id));
		
		// load parameters with tag
		loadParameters(nn, tag);
		
		// add updatelistener for tag
		addRepositoryListener(nn.id, tag);
	}
	
	public void nnDeploy(String name, int index, String tag){
		NeuralNetworkInstanceDTO nn = deploy(name, dianne.getRuntimes().get(index));
		
		// load parameters with tag
		loadParameters(nn, tag);
		
		// add updatelistener for tag
		addRepositoryListener(nn.id, tag);
	}
	

	private synchronized NeuralNetworkInstanceDTO deploy(String name, UUID runtimeId){
		try {
			NeuralNetworkInstanceDTO nn = dianne.deployNeuralNetwork(name, runtimeId);
			System.out.println("Deployed instance of "+nn.name+" ("+nn.id.toString()+")");
			return nn;
		} catch (InstantiationException e) {
			System.out.println("Error deploying instance of "+name);
			e.printStackTrace();
		}
		return null;
	}
	
	public void nnUndeploy(String nnId){
		NeuralNetworkInstanceDTO nn = dianne.getNeuralNetworkInstance(UUID.fromString(nnId));
		if(nn==null){
			System.out.println("No neural network deployed with id "+nnId);
			return;
		}
		undeploy(nn);
	}
	
	public void nnUndeploy(int index){
		List<NeuralNetworkInstanceDTO> nns = dianne.getNeuralNetworkInstances();
		if(index >= nns.size()){
			System.out.println("No neural network with index "+index);
			return;
		}
		NeuralNetworkInstanceDTO nn = nns.get(index);
		undeploy(nn);
	}
	
	private void undeploy(NeuralNetworkInstanceDTO nn){
		dianne.undeployNeuralNetwork(nn);
		
		ServiceRegistration r = repoListeners.get(nn.id);
		if(r!=null){
			r.unregister();
		}
	}
	
	public void sample(String dataset, String nnId, int sample, String...tags){

		Dataset d = datasets.get(dataset);
		if(d==null){
			System.out.println("Dataset "+dataset+" not available");
			return;
		}
		
		final int index = sample == -1 ? rand.nextInt(d.size()) : sample;
		
		NeuralNetwork nn = dianne.getNeuralNetwork(UUID.fromString(nnId));
		if(nn==null){
			System.out.println("Neural network "+nnId+" not available");
			return;
		}
	
		final String[] labels = nn.getOutputLabels();

		// get input and forward
		try {
			Tensor in = d.getInputSample(index);
			long t1 = System.currentTimeMillis();
			Tensor out = nn.forward(in, tags);
			long t2 = System.currentTimeMillis();
		
			int clazz = factory.getTensorMath().argmax(out);
			float max = factory.getTensorMath().max(out);
			String label = labels[clazz];
		
			System.out.println("Sample "+index+" (with tags "+Arrays.toString(tags)+") classified as: "+label+" (probability: "+max+")");
			System.out.println("Forward time: "+(t2-t1)+" ms");
		} catch(Exception e ){
			e.printStackTrace();
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
	
	private void loadParameters(NeuralNetworkInstanceDTO nn, String tag){
		Map<UUID, Tensor> parameters = repository.loadParameters(nn.name, tag);
		NeuralNetwork n = dianne.getNeuralNetwork(nn.id);
		if(n!=null){
			n.setParameters(parameters);
		}
	}
	
	private void addRepositoryListener(UUID nnId, String tag){
		ParameterUpdateListener listener = new ParameterUpdateListener(nnId);
		Dictionary<String, Object> props = new Hashtable<String, Object>();
		props.put("targets", new String[]{":"+tag});
		props.put("aiolos.unique", true);
		ServiceRegistration r = context.registerService(RepositoryListener.class, listener, props);
		repoListeners.put(nnId, r);
	}
	
	class ParameterUpdateListener implements RepositoryListener {

		final UUID nnId;
		
		public ParameterUpdateListener(UUID nnId) {
			this.nnId = nnId;
		}
		
		@Override
		public void onParametersUpdate(Collection<UUID> moduleIds,
				String... tag) {
			NeuralNetwork nn = dianne.getNeuralNetwork(nnId);
			if(nn!=null){
				moduleIds.stream().forEach(moduleId -> nn.setParameters(moduleId, repository.loadParameters(moduleId, tag)));
			}
		}
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
	
	@Reference
	void setDianneRepository(DianneRepository repo){
		this.repository = repo;
	}
	
	@Reference
	void setDianne(Dianne d){
		dianne = d;
	}

	@Reference
	void setTensorFactory(TensorFactory f){
		this.factory = f;
	}
}
