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
package be.iminds.iot.dianne.rnn.module.factory;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;

import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;

import be.iminds.iot.dianne.api.nn.module.AbstractModule;
import be.iminds.iot.dianne.api.nn.module.Module;
import be.iminds.iot.dianne.api.nn.module.dto.ModuleDTO;
import be.iminds.iot.dianne.api.nn.module.dto.ModulePropertyDTO;
import be.iminds.iot.dianne.api.nn.module.dto.ModuleTypeDTO;
import be.iminds.iot.dianne.api.nn.module.factory.ModuleFactory;
import be.iminds.iot.dianne.rnn.module.memory.SimpleMemory;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;

@Component(property={"aiolos.export=false"})
public class MemoryFactory implements ModuleFactory {

	private TensorFactory factory;
	
	private final Map<String, ModuleTypeDTO> supportedModules = new HashMap<String, ModuleTypeDTO>();
	
	@Activate
	void activate(){
		// build list of supported modules
		// TODO use reflection for this?
		
		addSupportedType( new ModuleTypeDTO("Simple Memory", "Memory", true, 
					new ModulePropertyDTO("Memory size", "size", Integer.class.getName())));
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
		
		// TODO use reflection for this?
		// for now just hard code an if/else for each known module
		String type = dto.type;
		UUID id = dto.id;

		switch(type){
		case "Simple Memory":
		{
			int size = Integer.parseInt(dto.properties.get("size"));
			
			if(parameters!=null){
				module = new SimpleMemory(factory, id, parameters);
			} else {
				module = new SimpleMemory(factory, id, size);
			}
			break;
		}
		default:
			throw new InstantiationException("Could not instantiate module of type "+type);
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
}
