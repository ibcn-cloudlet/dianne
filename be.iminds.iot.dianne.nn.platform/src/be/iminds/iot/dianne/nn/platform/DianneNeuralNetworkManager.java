package be.iminds.iot.dianne.nn.platform;

import java.io.IOException;
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
			UUID frameworkId) throws InstantiationException {
		return deployNeuralNetwork(name, frameworkId, new HashMap<UUID, UUID>());
	}

	@Override
	public NeuralNetworkInstanceDTO deployNeuralNetwork(String name,
			UUID frameworkId, Map<UUID, UUID> deployment) throws InstantiationException {
		
		NeuralNetworkDTO neuralNetwork = null;
		try {
			 neuralNetwork = repository.loadNeuralNetwork(name);
		} catch (IOException e) {
			throw new InstantiationException("Failed to deploy neural network "+name+": no such network");
		}
		
		UUID nnId = UUID.randomUUID();
		
		List<ModuleInstanceDTO> moduleInstances = new ArrayList<ModuleInstanceDTO>();
		for(ModuleDTO module : neuralNetwork.modules){
			UUID targetRuntime = deployment.get(module.id);
			if(targetRuntime==null){
				targetRuntime = frameworkId;
			}
			
			ModuleManager runtime = runtimes.get(targetRuntime);
			if(runtime==null){
				undeployNeuralNetwork(nnId);
				
				throw new InstantiationException("Failed to deploy modules to runtime "+targetRuntime+": no such runtime");
			}
			
			ModuleInstanceDTO instance = runtime.deployModule(module, nnId);
			moduleInstances.add(instance);
		}

		NeuralNetworkInstanceDTO nn = new NeuralNetworkInstanceDTO(nnId, name, moduleInstances);
		nns.put(nnId, nn);
		return nn;
	}

	@Override
	public void undeployNeuralNetwork(NeuralNetworkInstanceDTO nn) {
		nns.remove(nn.nnId);
		
		undeployNeuralNetwork(nn.nnId);
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
	public List<NeuralNetworkInstanceDTO> getNeuralNetworks() {
		List<NeuralNetworkInstanceDTO> list = new ArrayList<NeuralNetworkInstanceDTO>();
		list.addAll(nns.values());
		return list;
	}

	@Override
	public List<String> getSupportedNeuralNetworks() {
		return repository.avialableNeuralNetworks();
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
		runtimes.put(m.getFrameworkId(), m);
	}
	
	public void removeModuleManager(ModuleManager m, Map<String, Object> properties){
		runtimes.values().remove(m);
	}

}
