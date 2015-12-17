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
package be.iminds.iot.dianne.nn.module.composite;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.UUID;

import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;

import be.iminds.iot.dianne.api.nn.Dianne;
import be.iminds.iot.dianne.api.nn.NeuralNetwork;
import be.iminds.iot.dianne.api.nn.module.AbstractModule;
import be.iminds.iot.dianne.api.nn.module.Module;
import be.iminds.iot.dianne.api.nn.module.dto.ModuleDTO;
import be.iminds.iot.dianne.api.nn.module.dto.ModuleInstanceDTO;
import be.iminds.iot.dianne.api.nn.module.dto.ModuleTypeDTO;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkDTO;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkInstanceDTO;
import be.iminds.iot.dianne.api.nn.module.factory.ModuleFactory;
import be.iminds.iot.dianne.api.nn.runtime.DianneRuntime;
import be.iminds.iot.dianne.api.repository.DianneRepository;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;

@Component(property={"aiolos.export=false"})
public class CompositeModuleFactory implements ModuleFactory {

	private TensorFactory factory;
	
	private DianneRepository repository;

	private DianneRuntime runtime;
	
	private Dianne dianne;
	
	private final Map<String, ModuleTypeDTO> supportedModules = new HashMap<String, ModuleTypeDTO>();
	
	@Activate
	void activate(){
		// build list of supported modules
		// TODO use reflection for this?
		
		// which neural networks should be exposed as modules here?
		//addSupportedType( new ModuleTypeDTO("Linear", "Layer", true, 
		//			new ModulePropertyDTO("Input size", "input", Integer.class.getName()),
		//			new ModulePropertyDTO("Output size", "output", Integer.class.getName())));

		for(String nn : repository.availableNeuralNetworks()){
			// TODO which NNs to make available as composite modules
			// TODO which are the parameters? template in NeuralNetworkDTO? 
			addSupportedType( new ModuleTypeDTO(nn, "Composite", true ));
		}

	}
	
	
	@Override
	public Module createModule(ModuleDTO dto)
			throws InstantiationException {
		return createModule(dto, null);
	}
	
	@Override
	public Module createModule(ModuleDTO dto, Tensor parameters)
			throws InstantiationException {

		AbstractModule module = null;
		
		String nnName = dto.type;
		UUID id = dto.id;

		if(!supportedModules.containsKey(nnName)){
			throw new InstantiationException("Could not instantiate module of type "+nnName);
		}
		
		NeuralNetworkDTO nnDescription = repository.loadNeuralNetwork(nnName);

		// calculate parameters Tensor size
		// TODO should this somehow be made available by the module(dto) or something?
		// This is now replicated properties parsing code from the other factory
		// and size calculation from the module impl :-(
		int total = 0;
		int size = 0;
		LinkedHashMap<UUID, Integer> parameterMapping = new LinkedHashMap<>();
		for(ModuleDTO m : nnDescription.modules){
			switch(m.type){
			case "Linear":
				int inSize = Integer.parseInt(m.properties.get("input"));
				int outSize = Integer.parseInt(m.properties.get("output"));
				size = outSize*(inSize+1);
				parameterMapping.put(m.id, size);
				break;
			case "Convolution":
				int noInputPlanes = Integer.parseInt(m.properties.get("noInputPlanes"));
				int noOutputPlanes = Integer.parseInt(m.properties.get("noOutputPlanes"));
				int kernelWidth = Integer.parseInt(m.properties.get("kernelWidth"));
				int kernelHeight = Integer.parseInt(m.properties.get("kernelHeight"));
				
				size = noOutputPlanes*noInputPlanes*kernelWidth*kernelHeight+noOutputPlanes;
				parameterMapping.put(m.id, size);
				break;
			case "PReLU":
				size = 1;
				parameterMapping.put(m.id, size);
				break;
			default : 
				size = 0;
				break;
			}
			total+=size;
		}
		
		// narrow for each trainable module
		boolean hasParameters = true;
		if(parameters == null){
			parameters = factory.createTensor(total);
			parameters.fill(0.0f);
			hasParameters = false;
		} else {
			parameters.reshape(total);
		}
		
		Map<UUID, Tensor> narrowed = new HashMap<>();
		int offset = 0;
		Iterator<Entry<UUID, Integer>> it = parameterMapping.entrySet().iterator();
		while(it.hasNext()){
			Entry<UUID, Integer> e = it.next();
			int s = e.getValue();
			Tensor narrow = parameters.narrow(0, offset, s);
			narrowed.put(e.getKey(), narrow);
			
			// no parameters for composite found, load parameters for individual modules
			// useful for constructing ensembles
			if(!hasParameters){
				// try first with composite module id as tag
				try {
					Tensor t = repository.loadParameters(e.getKey(), id.toString());
					t.copyInto(narrow);
				} catch (Exception e1) {
					// try again without tag
					try {
						Tensor t = repository.loadParameters(e.getKey());
						t.copyInto(narrow);
					} catch (Exception e2) {
						// no paramters found
					}
				}
			}
			
			offset+=s;
		}
		
		// deploy each module and inject narrowed part of parameters
		Map<UUID, ModuleInstanceDTO> deployed = new HashMap<>();
		for(ModuleDTO m : nnDescription.modules){
			try {
				ModuleInstanceDTO mi = runtime.deployModule(m, id, narrowed.get(m.id));
				deployed.put(mi.moduleId, mi);
			} catch(Exception e){
				for(ModuleInstanceDTO mi : deployed.values()){
					runtime.undeployModule(mi);
				}
				throw new RuntimeException("Failed to deploy composite module "+id+": "+e.getMessage());
			}
		}
		NeuralNetworkInstanceDTO nnDTO = new NeuralNetworkInstanceDTO(id, nnName, deployed);
		
		// get NeuralNetwork object
		try {
			NeuralNetwork nn = dianne.getNeuralNetwork(nnDTO).getValue();
			
			// create CompositeModule with parameters and NeuralNetwork
			module = new CompositeModule(factory, id, parameters, nn, parameterMapping);
			
		} catch (Exception e) {
			throw new RuntimeException("Failed to deploy composite module "+id+": "+e.getMessage());
		} 

		return module;
	}

	@Override
	public List<ModuleTypeDTO> getAvailableModuleTypes() {
		return new ArrayList<ModuleTypeDTO>(supportedModules.values());
	}

	@Override
	public ModuleTypeDTO getModuleType(String name) {
		return supportedModules.get(name);
	}
	
	private boolean hasProperty(Map<String, String> config, String property){
		String value = (String) config.get(property);
		if(value==null){
			return false;
		} else if(value.isEmpty()){
			return false;
		}
		return true;
	}
	
	private void addSupportedType(ModuleTypeDTO t){
		supportedModules.put(t.type, t);
	}
	
	@Reference
	void setTensorFactory(TensorFactory f){
		this.factory = f;
	}
	
	@Reference
	void setDianneRepository(DianneRepository r){
		this.repository = r;
	}
	
	@Reference
	void setDianneRuntime(DianneRuntime r){
		this.runtime = r;
	}
	
	@Reference
	void setDianne(Dianne d){
		this.dianne = d;
	}
}
