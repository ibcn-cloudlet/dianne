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

import java.util.Map;
import java.util.UUID;

/**
 * Represents an instance of a neural network.
 * 
 * Contains the UUID of the nn instance, the name of the neural network,
 * and the list of ModuleInstances that this nn is composed of.
 * 
 * @author tverbele
 *
 */
public class NeuralNetworkInstanceDTO {

	// UUID of the neural network instance
	public final UUID id;
	
	// human readable neural network description
	public final String description;
	
	// neural network name
	public final String name;
	
	// optionally the complete neural network dto
	public NeuralNetworkDTO nn = null;
	
	// The list of ModuleInstances that this neural network instance is composed of
	public final Map<UUID, ModuleInstanceDTO> modules;
	
	public NeuralNetworkInstanceDTO(UUID id, String name, Map<UUID, ModuleInstanceDTO> modules){
		this(id, name, null, modules);
	}
	
	public NeuralNetworkInstanceDTO(UUID id, String name, String description, Map<UUID, ModuleInstanceDTO> modules){
		this.id = id;
		this.name = name;
		this.description = description;
		this.modules = modules;
	}
	
	public NeuralNetworkInstanceDTO(UUID id, NeuralNetworkDTO nn, Map<UUID, ModuleInstanceDTO> modules){
		this(id, nn, null, modules);
	}
	
	public NeuralNetworkInstanceDTO(UUID id, NeuralNetworkDTO nn, String description, Map<UUID, ModuleInstanceDTO> modules){
		this.id = id;
		this.name = nn.name;
		this.nn = nn;
		this.description = description;
		this.modules = modules;
	}
	
	@Override
	public boolean equals(Object o){
		if(!(o instanceof NeuralNetworkInstanceDTO)){
			return false;
		}
		
		NeuralNetworkInstanceDTO other = (NeuralNetworkInstanceDTO) o;
		return other.id.equals(id);
	}
	
	@Override
	public int hashCode(){
		return id.hashCode();
	}
}
