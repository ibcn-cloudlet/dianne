/*******************************************************************************
 * DIANNE  - Framework for distributed artificial neural networks
 * Copyright (C) 2015  iMinds - IBCN - UGent
 *
 * This file is part of DIANNE.
 *
 * DIANNE is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * Contributors:
 *     Tim Verbelen, Steven Bohez
 *******************************************************************************/
package be.iminds.iot.dianne.nn.platform;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

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
import be.iminds.iot.dianne.api.nn.platform.DiannePlatform;
import be.iminds.iot.dianne.api.nn.runtime.DianneRuntime;
import be.iminds.iot.dianne.api.repository.DianneRepository;

@Component
public class DiannePlatformImpl implements DiannePlatform {

	private DianneRepository repository;
	private Map<UUID, DianneRuntime> runtimes = new ConcurrentHashMap<UUID, DianneRuntime>();
	
	private Map<UUID, Map<UUID, Module>> modules = new ConcurrentHashMap<>();
	
	// available neural networks
	private Map<UUID, NeuralNetworkInstanceDTO> nnis = new ConcurrentHashMap<UUID, NeuralNetworkInstanceDTO>();

	private UUID frameworkId;

	// waiting set to wait until remote service available when migrating
	private Set<String> toWait = new HashSet<>();
	
	@Activate
	public void activate(BundleContext context) throws Exception {
		frameworkId = UUID.fromString(context.getProperty(Constants.FRAMEWORK_UUID));
	}
	

	@Override
	public NeuralNetworkInstanceDTO deployNeuralNetwork(String name, String... tags)
			throws InstantiationException {
		return deployNeuralNetwork(name, null, null, null, new HashMap<UUID, UUID>(), tags);
	}
	
	@Override
	public NeuralNetworkInstanceDTO deployNeuralNetwork(String name, Map<String, String> properties, String... tags)
			throws InstantiationException {
		return deployNeuralNetwork(name, null, properties, null, new HashMap<UUID, UUID>(), tags);
	}
	
	@Override
	public NeuralNetworkInstanceDTO deployNeuralNetwork(String name, String description, String... tags)
			throws InstantiationException {
		return deployNeuralNetwork(name, description, null, null, new HashMap<UUID, UUID>(), tags);
	}
	
	@Override
	public NeuralNetworkInstanceDTO deployNeuralNetwork(String name, String description, Map<String, String> properties,
			String... tags) throws InstantiationException {
		return deployNeuralNetwork(name, description, properties, null, new HashMap<UUID, UUID>(), tags);
	}

	@Override
	public NeuralNetworkInstanceDTO deployNeuralNetwork(String name, 
			UUID runtimeId, String... tags) throws InstantiationException {
		return deployNeuralNetwork(name, null, null, runtimeId, new HashMap<UUID, UUID>(), tags);
	}

	@Override
	public NeuralNetworkInstanceDTO deployNeuralNetwork(String name, Map<String, String> properties,
			UUID runtimeId, String... tags) throws InstantiationException {
		return deployNeuralNetwork(name, null, properties, runtimeId, new HashMap<UUID, UUID>(), tags);
	}
	
	@Override
	public NeuralNetworkInstanceDTO deployNeuralNetwork(String name, String description,
			UUID runtimeId, Map<UUID, UUID> deployment, String... tags) throws InstantiationException {
		return deployNeuralNetwork(name, description, null, runtimeId, deployment, tags);
	}

	@Override
	public NeuralNetworkInstanceDTO deployNeuralNetwork(String name, String description, UUID runtimeId, String... tags)
			throws InstantiationException {
		return deployNeuralNetwork(name, description, null, runtimeId, new HashMap<UUID, UUID>(), tags);
	}

	@Override
	public NeuralNetworkInstanceDTO deployNeuralNetwork(String name, String description, Map<String, String> properties,
			UUID runtimeId, String... tags) throws InstantiationException {
		return deployNeuralNetwork(name, description, properties, runtimeId, new HashMap<UUID, UUID>(), tags);
	}

	@Override
	public NeuralNetworkInstanceDTO deployNeuralNetwork(String name, UUID runtimeId, Map<UUID, UUID> deployment,
			String... tags) throws InstantiationException {
		return deployNeuralNetwork(name, null, null, runtimeId, deployment, tags);
	}

	@Override
	public NeuralNetworkInstanceDTO deployNeuralNetwork(String name, Map<String, String> properties, UUID runtimeId,
			Map<UUID, UUID> deployment, String... tags) throws InstantiationException {
		return deployNeuralNetwork(name, null, properties, runtimeId, deployment, tags);
	}
	
	@Override
	public NeuralNetworkInstanceDTO deployNeuralNetwork(String name, String description, Map<String, String> properties,
			UUID runtimeId, Map<UUID, UUID> deployment, String... tags) throws InstantiationException {
		
		NeuralNetworkDTO neuralNetwork = null;
		try {
			 neuralNetwork = repository.loadNeuralNetwork(name);
		} catch (Exception e) {
			throw new InstantiationException("Failed to deploy neural network "+name+": no such network");
		}
		
		UUID nnId = UUID.randomUUID();
		
		Map<UUID, ModuleInstanceDTO> moduleInstances = new HashMap<UUID, ModuleInstanceDTO>();
		for(ModuleDTO module : neuralNetwork.modules.values()){
			mergeProperties(module, properties);
			
			UUID targetRuntime = deployment.get(module.id);
			if(targetRuntime==null){
				if(runtimeId == null){
					if(!runtimes.containsKey(frameworkId)){
						throw new InstantiationException("No local runtime available");
					}
					runtimeId = frameworkId;
				}
				
				targetRuntime = runtimeId;
			}
			
			DianneRuntime runtime = runtimes.get(targetRuntime);
			if(runtime==null){
				undeployNeuralNetwork(nnId);
				
				throw new InstantiationException("Failed to deploy modules to runtime "+targetRuntime+": no such runtime");
			}
			
			ModuleInstanceDTO instance = runtime.deployModule(module, nnId, tags);
			moduleInstances.put(instance.moduleId, instance);
		}

		NeuralNetworkInstanceDTO nni = new NeuralNetworkInstanceDTO(nnId, name, description, moduleInstances);
		nnis.put(nnId, nni);
		return nni;
	}
	
	
	@Override
	public NeuralNetworkInstanceDTO deployNeuralNetwork(NeuralNetworkDTO nn, String... tags)
			throws InstantiationException {
		repository.storeNeuralNetwork(nn);
		return deployNeuralNetwork(nn.name, tags);
	}
	
	@Override
	public NeuralNetworkInstanceDTO deployNeuralNetwork(NeuralNetworkDTO nn, String description, String... tags)
			throws InstantiationException {
		repository.storeNeuralNetwork(nn);
		return deployNeuralNetwork(nn.name, description, tags);
	}
	
	@Override
	public NeuralNetworkInstanceDTO deployNeuralNetwork(NeuralNetworkDTO nn, 
			UUID runtimeId, String... tags) throws InstantiationException {
		repository.storeNeuralNetwork(nn);
		return deployNeuralNetwork(nn.name, (String)null, runtimeId, new HashMap<UUID, UUID>(), tags);
	}
	
	@Override
	public NeuralNetworkInstanceDTO deployNeuralNetwork(NeuralNetworkDTO nn, String description,
			UUID runtimeId, String... tags) throws InstantiationException {
		repository.storeNeuralNetwork(nn);
		return deployNeuralNetwork(nn.name, description, runtimeId, new HashMap<UUID, UUID>(), tags);
	}


	@Override
	public NeuralNetworkInstanceDTO deployNeuralNetwork(NeuralNetworkDTO nn,
			UUID runtimeId,  Map<UUID, UUID> deployment, String... tags) throws InstantiationException {
		repository.storeNeuralNetwork(nn);
		return deployNeuralNetwork(nn.name, (String)null, runtimeId, deployment, tags);
	}
	
	@Override
	public NeuralNetworkInstanceDTO deployNeuralNetwork(NeuralNetworkDTO nn, String description,
			UUID runtimeId, Map<UUID, UUID> deployment, String... tags) throws InstantiationException {
		repository.storeNeuralNetwork(nn);
		return deployNeuralNetwork(nn.name, description, runtimeId, deployment);
	}

	@Override
	public void undeployNeuralNetwork(NeuralNetworkInstanceDTO nni) {
		undeployNeuralNetwork(nni.id);
	}

	@Override
	public void undeployNeuralNetwork(UUID nnId){
		// undeploy all modules with nnId
		synchronized(runtimes){
			for(DianneRuntime runtime : runtimes.values()){
				runtime.undeployModules(nnId);
			}
		}
		
		// remove all modules from nni
		if(nnis.containsKey(nnId)){
			nnis.get(nnId).modules.clear();
			nnis.remove(nnId);
		}
	}
	
	@Override
	public List<ModuleInstanceDTO> deployModules(UUID nnId,
			List<ModuleDTO> modules, UUID runtimeId, String... tags) throws InstantiationException{
		return deployModules(nnId, null, null, modules, runtimeId, tags);
	}
	
	@Override
	public List<ModuleInstanceDTO> deployModules(UUID nnId, String name,
			List<ModuleDTO> modules, UUID runtimeId, String... tags) throws InstantiationException{
		return deployModules(nnId, name, null, modules, runtimeId, tags);
	}
	
	@Override
	public List<ModuleInstanceDTO> deployModules(UUID nnId, String name, String description,
			List<ModuleDTO> modules, UUID runtimeId, String... tags)
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
		
		if(runtimeId == null){
			if(!runtimes.containsKey(frameworkId)){
				throw new InstantiationException("No local runtime available");
			}
			runtimeId = frameworkId;
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
			
			// if migrate - copy parameters from old
			// TODO should we copy parameters or rather load from repo
			//Tensor parameters = null;
			//if(old!=null){
			//	parameters = runtimes.get(old.runtimeId).getModuleParameters(old);
			//}
			if(old!=null){
				synchronized(toWait){
					toWait.add(nnId+":"+old.moduleId);
				}
			}

			ModuleInstanceDTO moduleInstance = runtime.deployModule(module, nnId, tags);
			
			// migrate - undeploy old
			if(old!=null){
				while(toWait.contains(nnId+":"+old.moduleId)){
					synchronized(toWait){
						if(toWait.contains(nnId+":"+old.moduleId)){
							try {
								// TODO add timeout and throw exception?!
								toWait.wait();
							} catch (InterruptedException e) {
							}
						}
					}
				}
				
				undeployModules(Collections.singletonList(old));
			}
			
			
			// put in NeuralNetworkInstance DTO
			nni.modules.put(moduleInstance.moduleId, moduleInstance);
			moduleInstances.add(moduleInstance);
		}
		
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
	public List<String> getAvailableNeuralNetworks() {
		return repository.availableNeuralNetworks();
	}
	
	@Override
	public NeuralNetworkDTO getAvailableNeuralNetwork(String name) {
		return repository.loadNeuralNetwork(name);
	}
	
	@Override
	public Map<UUID, String> getRuntimes() {
		/* TODO this will invoke a (remote) call to each runtime each time runtimes are fetched
		 This is not optimal ... should be handled better with for example service property.
		 Change this once we move to R6 and use configurer for this? */ 
		Map<UUID, String> result = runtimes.entrySet().stream().collect(Collectors.toMap(e -> e.getKey(), e -> e.getValue().getRuntimeName()));
		return result;
	}
	
	@Reference
	void setDianneRepository(DianneRepository r){
		repository = r;
	}
	
	@Reference(cardinality=ReferenceCardinality.MULTIPLE, 
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

	void updatedModule(Module m, Map<String, Object> properties){
		UUID moduleId = UUID.fromString((String)properties.get("module.id"));
		UUID nnId = UUID.fromString((String)properties.get("nn.id"));
		
		synchronized(toWait){
			toWait.remove(nnId+":"+moduleId);
			toWait.notifyAll();
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

	private void mergeProperties(ModuleDTO module, Map<String, String> properties){
		if(properties == null)
			return;
		
		properties.entrySet().stream()
			.forEach(e -> {
				int index = e.getKey().indexOf('.');
				if(index < 0 )
					return;
				
				// scan for property keys of the form   some-id.some-key
				String id = e.getKey().substring(0, index);
				String key = e.getKey().substring(index+1);
				
				if(module.id.toString().equals(id)
					|| id.equals(module.properties.get("name"))
					|| id.equals(module.type)){
					module.properties.put(key, e.getValue());
				}
			});
	}
}
