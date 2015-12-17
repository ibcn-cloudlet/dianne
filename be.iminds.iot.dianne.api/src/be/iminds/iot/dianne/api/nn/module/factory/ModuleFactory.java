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
package be.iminds.iot.dianne.api.nn.module.factory;

import java.util.List;

import be.iminds.iot.dianne.api.nn.module.Module;
import be.iminds.iot.dianne.api.nn.module.dto.ModuleDTO;
import be.iminds.iot.dianne.api.nn.module.dto.ModuleTypeDTO;
import be.iminds.iot.dianne.tensor.Tensor;

/**
 * A ModuleFactory knows how to create Module instances from a ModuleDTO.
 * 
 * The factory also announces all ModuleTypes it knows to construct.
 * @author tverbele
 *
 */
public interface ModuleFactory {

	/**
	 * Create a new Module instance from a ModuleDTO
	 * @param dto the dto describing the module to construct
	 * @return the constructed Module
	 * @throws InstantiationException thrown when it failed to instantiate the Module
	 */
	Module createModule(ModuleDTO dto) throws InstantiationException;
	
	/**
	 * Create a new Module instance from a ModuleDTO, if this is a trainable module parameters
	 * can be passed here for construction
	 * @param dto the dto describing the module to construct
	 * @param parameters parameters to pass for module construction
	 * @return the constructed module
	 * @throws InstantiationException
	 */
	Module createModule(ModuleDTO dto, Tensor parameters) throws InstantiationException;
	
	/**
	 * Get the list of module types that this factory can construct
	 * 
	 * @return available module types
	 */
	List<ModuleTypeDTO> getAvailableModuleTypes();
	
	/**
	 * Get a detailed ModuleTypeDTO for a given module type name
	 * 
	 * @param name name of the module type
	 * @return the detailed ModuleDTO matching this type, or null if this type is not available
	 */
	ModuleTypeDTO getModuleType(String name);
}
