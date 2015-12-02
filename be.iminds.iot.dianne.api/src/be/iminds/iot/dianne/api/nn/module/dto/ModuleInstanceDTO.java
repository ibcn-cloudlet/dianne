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
package be.iminds.iot.dianne.api.nn.module.dto;

import java.util.UUID;

/**
 * Represents an actual instance of a Neural Network module
 * 
 * Uniquely identified by the moduleId (which module is it), the runtimeId 
 * (where is it deployed) and nnId (which neural network instance does it belong to)
 * 
 * @author tverbele
 *
 */
public class ModuleInstanceDTO {

	// Module UUID of this module
	public final UUID moduleId;
	
	// UUID of the Neural Network this instance belongs to
	public final UUID nnId;
	
	// UUID of the runtime where the module instance is deployed
	public final UUID runtimeId;

	// Module type
	public final ModuleDTO module;
	
	public ModuleInstanceDTO(ModuleDTO module, UUID nnId, UUID runtimeId){
		this.moduleId = module.id;
		this.nnId = nnId;
		this.runtimeId = runtimeId;
		this.module = module;
	}
	
	@Override
	public boolean equals(Object o){
		if(!(o instanceof ModuleInstanceDTO)){
			return false;
		}
		
		ModuleInstanceDTO other = (ModuleInstanceDTO) o;
		return other.moduleId.equals(moduleId)
				&&	other.nnId.equals(nnId)
				&&  other.runtimeId.equals(runtimeId);
	}
	
	@Override
	public int hashCode(){
		return moduleId.hashCode() + 31*nnId.hashCode() + runtimeId.hashCode();
	}
}
