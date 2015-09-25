package be.iminds.iot.dianne.nn.platform;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;

import org.osgi.framework.BundleContext;
import org.osgi.framework.Constants;
import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;
import org.osgi.service.component.annotations.ReferenceCardinality;
import org.osgi.service.component.annotations.ReferencePolicy;

import be.iminds.iot.dianne.api.nn.module.Module;
import be.iminds.iot.dianne.api.nn.module.dto.ModuleDTO;
import be.iminds.iot.dianne.api.nn.module.dto.ModuleInstanceDTO;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkDTO;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkInstanceDTO;
import be.iminds.iot.dianne.api.nn.platform.Dianne;
import be.iminds.iot.dianne.api.nn.platform.NeuralNetwork;
import be.iminds.iot.dianne.api.nn.runtime.DianneRuntime;
import be.iminds.iot.dianne.api.repository.DianneRepository;
import be.iminds.iot.dianne.tensor.TensorFactory;

@Component(property={"aiolos.export=false"})
public class DianneImpl implements Dianne {

	private TensorFactory factory;
	private DianneRepository repository;
	private Map<UUID, DianneRuntime> runtimes = Collections.synchronizedMap(new HashMap<UUID, DianneRuntime>());
	
	private Map<UUID, Map<UUID, Module>> modules = Collections.synchronizedMap(new HashMap<>());
	
	// available neural networks
	private Map<UUID, NeuralNetworkInstanceDTO> nnis = Collections.synchronizedMap(new HashMap<UUID, NeuralNetworkInstanceDTO>());
	private Map<UUID, NeuralNetworkWrapper> nns = Collections.synchronizedMap(new HashMap<UUID, NeuralNetworkWrapper>());
	private Map<UUID, NeuralNetwork> nnServices = Collections.synchronizedMap(new HashMap<UUID, NeuralNetwork>());
	
	private UUID frameworkId;
	private BundleContext context;
	
	@Activate
	public void activate(BundleContext context) throws Exception {
		frameworkId = UUID.fromString(context.getProperty(Constants.FRAMEWORK_UUID));
		this.context = context;
	}
	

	@Override
	public NeuralNetworkInstanceDTO deployNeuralNetwork(String name)
			throws InstantiationException {
		if(!runtimes.containsKey(frameworkId)){
			throw new InstantiationException("No local runtime available");
		}
		return deployNeuralNetwork(name, null, frameworkId, new HashMap<UUID, UUID>());
	}
	
	@Override
	public NeuralNetworkInstanceDTO deployNeuralNetwork(String name, String description)
			throws InstantiationException {
		if(!runtimes.containsKey(frameworkId)){
			throw new InstantiationException("No local runtime available");
		}
		return deployNeuralNetwork(name, description, frameworkId, new HashMap<UUID, UUID>());
	}
	
	@Override
	public NeuralNetworkInstanceDTO deployNeuralNetwork(String name, 
			UUID runtimeId) throws InstantiationException {
		return deployNeuralNetwork(name, null, runtimeId, new HashMap<UUID, UUID>());
	}
	
	@Override
	public NeuralNetworkInstanceDTO deployNeuralNetwork(String name, String description,
			UUID runtimeId) throws InstantiationException {
		return deployNeuralNetwork(name, description, runtimeId, new HashMap<UUID, UUID>());
	}


	@Override
	public NeuralNetworkInstanceDTO deployNeuralNetwork(String name,
			UUID runtimeId,  Map<UUID, UUID> deployment) throws InstantiationException {
		return deployNeuralNetwork(name, null, runtimeId, deployment);
	}
	
	@Override
	public NeuralNetworkInstanceDTO deployNeuralNetwork(String name, String description,
			UUID runtimeId, Map<UUID, UUID> deployment) throws InstantiationException {
		
		NeuralNetworkDTO neuralNetwork = null;
		try {
			 neuralNetwork = repository.loadNeuralNetwork(name);
		} catch (Exception e) {
			throw new InstantiationException("Failed to deploy neural network "+name+": no such network");
		}
		
		UUID nnId = UUID.randomUUID();
		
		Map<UUID, ModuleInstanceDTO> moduleInstances = new HashMap<UUID, ModuleInstanceDTO>();
		for(ModuleDTO module : neuralNetwork.modules){
			UUID targetRuntime = deployment.get(module.id);
			if(targetRuntime==null){
				targetRuntime = runtimeId;
			}
			
			DianneRuntime runtime = runtimes.get(targetRuntime);
			if(runtime==null){
				undeployNeuralNetwork(nnId);
				
				throw new InstantiationException("Failed to deploy modules to runtime "+targetRuntime+": no such runtime");
			}
			
			ModuleInstanceDTO instance = runtime.deployModule(module, nnId);
			moduleInstances.put(instance.moduleId, instance);
		}

		NeuralNetworkInstanceDTO nni = new NeuralNetworkInstanceDTO(nnId, name, description, moduleInstances);
		nnis.put(nnId, nni);
		
		updateNeuralNetwork(nnId);
		
		return nni;
	}

	@Override
	public void undeployNeuralNetwork(NeuralNetworkInstanceDTO nni) {
		undeployNeuralNetwork(nni.id);
	
		// remove all modules from nni
		nnis.get(nni.id).modules.clear();
		
		updateNeuralNetwork(nni.id);
	
		nnis.remove(nni.id);
	}

	private void undeployNeuralNetwork(UUID nnId){
		// undeploy all modules with nnId
		synchronized(runtimes){
			for(DianneRuntime runtime : runtimes.values()){
				runtime.undeployModules(nnId);
			}
		}
	}
	
	@Override
	public List<ModuleInstanceDTO> deployModules(UUID nnId,
			List<ModuleDTO> modules, UUID runtimeId) throws InstantiationException{
		return deployModules(nnId, null, null, modules, runtimeId);
	}
	
	@Override
	public List<ModuleInstanceDTO> deployModules(UUID nnId, String name,
			List<ModuleDTO> modules, UUID runtimeId) throws InstantiationException{
		return deployModules(nnId, name, null, modules, runtimeId);
	}
	
	@Override
	public List<ModuleInstanceDTO> deployModules(UUID nnId, String name, String description,
			List<ModuleDTO> modules, UUID runtimeId)
			throws InstantiationException {
		List<ModuleInstanceDTO> moduleInstances = new ArrayList<ModuleInstanceDTO>();

		if(nnId==null){
			nnId = UUID.randomUUID();
		}
		
		// if neural network instance already exists, update nn DTO and migrate modules if already deployed somewhere else
		NeuralNetworkInstanceDTO nni = nnis.get(nnId);
		if(nni==null){
			nni = new NeuralNetworkInstanceDTO(nnId, name, description, new HashMap<UUID, ModuleInstanceDTO>());
			nnis.put(nnId, nni);
		}
		
		DianneRuntime runtime = runtimes.get(runtimeId);
		if(runtime==null){
			throw new InstantiationException("Failed to deploy modules to runtime "+runtimeId+": no such runtime");
		}
		
		for(ModuleDTO module : modules){
			ModuleInstanceDTO old = null;
			if(nni!=null){
				old = nni.modules.get(module.id);
			}
			
			if(old!=null && old.runtimeId.equals(runtimeId)){
				// already deployed on target runtime ...
				continue;
			}
			
			ModuleInstanceDTO moduleInstance = runtime.deployModule(module, nnId);
			
			// put in NeuralNetworkInstance DTO
			nni.modules.put(moduleInstance.moduleId, moduleInstance);
			
			// migrate - undeploy old
			if(old!=null){
				undeployModules(Collections.singletonList(old));
			}
			
			moduleInstances.add(moduleInstance);
		}
		
		updateNeuralNetwork(nnId);
		
		return moduleInstances;
	}

	@Override
	public void undeployModules(List<ModuleInstanceDTO> moduleInstances) {
		Set<UUID> nnIds = new HashSet<UUID>();
		for(ModuleInstanceDTO moduleInstance : moduleInstances){
			nnIds.add(moduleInstance.nnId);
			nnis.get(moduleInstance.nnId).modules.remove(moduleInstance.moduleId);
			
			DianneRuntime runtime = runtimes.get(moduleInstance.runtimeId);
			if(runtime!=null){
				runtime.undeployModule(moduleInstance);
			}
		}
		for(UUID nnId : nnIds){
			updateNeuralNetwork(nnId);
		}
		for(UUID nnId : nnIds){
			NeuralNetworkInstanceDTO nn = nnis.get(nnId);
			if(nn!=null && nn.modules.size()==0){
				nnis.remove(nnId);
			}
		}
	}
	
	@Override
	public List<NeuralNetworkInstanceDTO> getNeuralNetworkInstances() {
		List<NeuralNetworkInstanceDTO> list = new ArrayList<NeuralNetworkInstanceDTO>();
		list.addAll(nnis.values());
		return list;
	}


	@Override
	public NeuralNetworkInstanceDTO getNeuralNetworkInstance(UUID nnId) {
		return nnis.get(nnId);
	}
	
	@Override
	public List<String> getSupportedNeuralNetworks() {
		return repository.availableNeuralNetworks();
	}
	
	@Override
	public List<UUID> getRuntimes() {
		List<UUID> list = new ArrayList<UUID>();
		list.addAll(runtimes.keySet());
		return list;
	}
	
	@Reference
	void setDianneRepository(DianneRepository r){
		repository = r;
	}
	
	@Reference
	void setTensorFactory(TensorFactory f){
		factory = f;
	}

	@Reference(cardinality=ReferenceCardinality.AT_LEAST_ONE, 
			policy=ReferencePolicy.DYNAMIC)
	void addDianneRuntime(DianneRuntime r, Map<String, Object> properties){
		runtimes.put(r.getRuntimeId(), r);
	}
	
	void removeDianneRuntime(DianneRuntime r, Map<String, Object> properties){
		runtimes.values().remove(r);
	}

	@Reference(cardinality=ReferenceCardinality.MULTIPLE, 
			policy=ReferencePolicy.DYNAMIC)
	void addModule(Module m, Map<String, Object> properties){
		UUID moduleId = UUID.fromString((String)properties.get("module.id"));
		UUID nnId = UUID.fromString((String)properties.get("nn.id"));
	
		synchronized(modules){
			Map<UUID, Module> nnm = modules.get(nnId);
			if(nnm==null){
				nnm = new HashMap<>();
				modules.put(nnId, nnm);
			}
			nnm.put(moduleId, m);
		}
	}
	
	void removeModule(Module m, Map<String, Object> properties){
		UUID moduleId = UUID.fromString((String)properties.get("module.id"));
		UUID nnId = UUID.fromString((String)properties.get("nn.id"));
	
		synchronized(modules){
			Map<UUID, Module> nnm = modules.get(nnId);
			if(nnm != null){ // should not be null?
				nnm.remove(moduleId);
				
				if(nnm.size() == 0){
					modules.remove(nnId);
				}
			}
		}
	}
	
	// use separte service tracker in order to have the actual service from the registry
	// instead of the NeuralNetworkWrapper directly
	@Override
	public NeuralNetwork getNeuralNetwork(UUID nnId) {
		NeuralNetwork nn = nnServices.get(nnId);
		if(nn==null){
			System.err.println("No neural network found with id "+nnId);
		}
		return nn;
	}
	
	@Reference(cardinality=ReferenceCardinality.MULTIPLE, 
			policy=ReferencePolicy.DYNAMIC)
	public void addNeuralNetwork(NeuralNetwork nn, Map<String, Object> properties){
		String nnId = (String)properties.get("nn.id");
		nnServices.put(UUID.fromString(nnId), nn);
	}
	
	public void removeNeuralNetwork(NeuralNetwork nn, Map<String, Object> properties){
		String nnId = (String)properties.get("nn.id");
		nnServices.remove(UUID.fromString(nnId));
	}
	
	private void updateNeuralNetwork(UUID nnId){
		NeuralNetworkInstanceDTO nni = nnis.get(nnId);
		if(nni == null){
			return;
		}
		
		NeuralNetworkDTO template = repository.loadNeuralNetwork(nni.name);
		if(nni.modules.size()!= template.modules.size()){
			// not all modules deployed ... remove and unregister NN
			NeuralNetworkWrapper nn = nns.remove(nni.id);
			if(nn!=null){
				nn.unregister();
			}
		} else {
			// create/update wrapper and register
			Collection<Module> m = modules.get(nni.id).values();
					
			NeuralNetworkWrapper nn = nns.get(nni.id);
			if(nn!=null){
				nn.setModules(m);
			} else {
				nn = new NeuralNetworkWrapper(nni, m, repository, factory, context);
				nns.put(nni.id, nn);
				nn.register();
			}
		}
		
	}
}
