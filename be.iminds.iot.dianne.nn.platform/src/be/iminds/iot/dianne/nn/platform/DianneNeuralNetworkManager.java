package be.iminds.iot.dianne.nn.platform;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;

import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;
import org.osgi.service.component.annotations.ReferenceCardinality;
import org.osgi.service.component.annotations.ReferencePolicy;

import be.iminds.iot.dianne.api.nn.module.dto.ModuleDTO;
import be.iminds.iot.dianne.api.nn.module.dto.ModuleInstanceDTO;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkDTO;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkInstanceDTO;
import be.iminds.iot.dianne.api.nn.platform.NeuralNetworkManager;
import be.iminds.iot.dianne.api.nn.runtime.ModuleManager;
import be.iminds.iot.dianne.api.repository.DianneRepository;

@Component
public class DianneNeuralNetworkManager implements NeuralNetworkManager {

	private DianneRepository repository;
	private Map<UUID, ModuleManager> runtimes = Collections.synchronizedMap(new HashMap<UUID, ModuleManager>());

	private Map<UUID, NeuralNetworkInstanceDTO> nns = Collections.synchronizedMap(new HashMap<UUID, NeuralNetworkInstanceDTO>());
	
	@Override
	public NeuralNetworkInstanceDTO deployNeuralNetwork(String name,
			UUID runtimeId) throws InstantiationException {
		return deployNeuralNetwork(name, runtimeId, new HashMap<UUID, UUID>());
	}

	@Override
	public NeuralNetworkInstanceDTO deployNeuralNetwork(String name,
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
			
			ModuleManager runtime = runtimes.get(targetRuntime);
			if(runtime==null){
				undeployNeuralNetwork(nnId);
				
				throw new InstantiationException("Failed to deploy modules to runtime "+targetRuntime+": no such runtime");
			}
			
			ModuleInstanceDTO instance = runtime.deployModule(module, nnId);
			moduleInstances.put(instance.moduleId, instance);
		}

		NeuralNetworkInstanceDTO nn = new NeuralNetworkInstanceDTO(nnId, name, moduleInstances);
		nns.put(nnId, nn);
		return nn;
	}

	@Override
	public void undeployNeuralNetwork(NeuralNetworkInstanceDTO nn) {
		nns.remove(nn.id);
		
		undeployNeuralNetwork(nn.id);
	}

	private void undeployNeuralNetwork(UUID nnId){
		// undeploy all modules with nnId
		synchronized(runtimes){
			for(ModuleManager runtime : runtimes.values()){
				runtime.undeployModules(nnId);
			}
		}
	}
	
	@Override
	public List<ModuleInstanceDTO> deployModules(UUID nnId,
			List<ModuleDTO> modules, UUID runtimeId)
			throws InstantiationException {
		List<ModuleInstanceDTO> moduleInstances = new ArrayList<ModuleInstanceDTO>();

		NeuralNetworkInstanceDTO nn = null; 
		if(nnId==null){
			nnId = UUID.randomUUID();
		} else {
			// if neural network instance already exists, update nn DTO and migrate modules if already deployed somewhere else
			nn = nns.get(nnId);
		}
		
		ModuleManager runtime = runtimes.get(runtimeId);
		if(runtime==null){
			throw new InstantiationException("Failed to deploy modules to runtime "+runtimeId+": no such runtime");
		}
		
		for(ModuleDTO module : modules){
			ModuleInstanceDTO old = null;
			if(nn!=null){
				old = nn.modules.get(module.id);
			}
			
			if(old!=null && old.runtimeId.equals(runtimeId)){
				// already deployed on target runtime ...
				continue;
			}
			
			ModuleInstanceDTO moduleInstance = runtime.deployModule(module, nnId);
			
			// replace in NeuralNetworkInstance DTO
			if(nn!=null){
				nn.modules.put(moduleInstance.moduleId, moduleInstance);
			}
			
			// migrate - undeploy old
			if(old!=null){
				undeployModules(Collections.singletonList(old));
			}
		}
		
		return moduleInstances;
	}

	@Override
	public void undeployModules(List<ModuleInstanceDTO> moduleInstances) {
		for(ModuleInstanceDTO moduleInstance : moduleInstances){
			ModuleManager runtime = runtimes.get(moduleInstance.runtimeId);
			if(runtime!=null){
				runtime.undeployModule(moduleInstance);
			}
		}
	}
	
	@Override
	public List<NeuralNetworkInstanceDTO> getNeuralNetworks() {
		List<NeuralNetworkInstanceDTO> list = new ArrayList<NeuralNetworkInstanceDTO>();
		list.addAll(nns.values());
		return list;
	}


	@Override
	public NeuralNetworkInstanceDTO getNeuralNetwork(UUID nnId) {
		return nns.get(nnId);
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
	public void setDianneRepository(DianneRepository r){
		repository = r;
	}

	@Reference(cardinality=ReferenceCardinality.AT_LEAST_ONE, 
			policy=ReferencePolicy.DYNAMIC)
	public void addModuleManager(ModuleManager m, Map<String, Object> properties){
		runtimes.put(m.getRuntimeId(), m);
	}
	
	public void removeModuleManager(ModuleManager m, Map<String, Object> properties){
		runtimes.values().remove(m);
	}

}
