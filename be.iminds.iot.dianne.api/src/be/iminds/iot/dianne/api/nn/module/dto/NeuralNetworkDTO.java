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

import java.util.List;

/**
 * Represents a Neural Network being a list of ModuleDTOs and a name.
 * 
 * An instance of this network can be deployed by deploying an instance of
 * each of the modules and connecting them together.
 *  
 * @author tverbele
 *
 */
public class NeuralNetworkDTO {

	// name identifier of this neural network
	public final String name;
	
	// the ModuleDTOs that this neural network consists of
	public final List<ModuleDTO> modules;
	
	
	public NeuralNetworkDTO(String name, List<ModuleDTO> modules){
		this.name = name;
		this.modules = modules;
	}
	
	@Override
	public boolean equals(Object o){
		if(!(o instanceof NeuralNetworkDTO)){
			return false;
		}
		
		NeuralNetworkDTO other = (NeuralNetworkDTO) o;
		return other.name.equals(name);
	}
	
	@Override
	public int hashCode(){
		return name.hashCode();
	}
}
