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
package be.iminds.iot.dianne.rl.module.factory;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;

import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Component;

import be.iminds.iot.dianne.api.nn.module.AbstractModule;
import be.iminds.iot.dianne.api.nn.module.Module;
import be.iminds.iot.dianne.api.nn.module.dto.ModuleDTO;
import be.iminds.iot.dianne.api.nn.module.dto.ModulePropertyDTO;
import be.iminds.iot.dianne.api.nn.module.dto.ModuleTypeDTO;
import be.iminds.iot.dianne.api.nn.module.factory.ModuleFactory;
import be.iminds.iot.dianne.api.nn.module.factory.ModuleTypeNotSupportedException;
import be.iminds.iot.dianne.rl.module.DuelJoin;
import be.iminds.iot.dianne.rl.module.NormalizedAdvantageFunction;
import be.iminds.iot.dianne.tensor.Tensor;

@Component(property={"aiolos.export=false"})
public class RLModuleFactory implements ModuleFactory {

	private final Map<String, ModuleTypeDTO> supportedModules = new HashMap<String, ModuleTypeDTO>();
	
	@Activate
	void activate(){
		// build list of supported modules
		// TODO use reflection for this?
		addSupportedType( new ModuleTypeDTO("DuelJoin", "Join", false));
		addSupportedType(new ModuleTypeDTO("NormalizedAdvantageFunction", "Layer", false,
				new ModulePropertyDTO("Action dimensions", "actionDims", Integer.class.getName())));
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
		case "DuelJoin":
		{
			module = new DuelJoin(id);
			break;
		}
		case "NormalizedAdvantageFunction":
		{
			module = new NormalizedAdvantageFunction(id, Integer.parseInt(dto.properties.get("actionDims")));
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
	
	@Override
	public int parameterSize(ModuleDTO m) throws ModuleTypeNotSupportedException {
		if(!supportedModules.containsKey(m.type))
			throw new ModuleTypeNotSupportedException(m.type);
		return 0;
	}

	@Override
	public int memorySize(ModuleDTO m) throws ModuleTypeNotSupportedException {
		if(!supportedModules.containsKey(m.type))
			throw new ModuleTypeNotSupportedException(m.type);
		return 0;
	}
	
	private void addSupportedType(ModuleTypeDTO t){
		supportedModules.put(t.type, t);
	}
	
}
