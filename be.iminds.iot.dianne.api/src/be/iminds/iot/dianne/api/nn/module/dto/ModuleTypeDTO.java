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

/**
 * A ModuleType declares how a certain module should look like.
 * 
 * The type String is used to identify the ModuleType and is used in the ModuleDTO
 * to determine the type. This DTO can be used for a UI builder to discover the properties
 * that should be set to create an instance of such a Module type.
 * 
 * @author tverbele
 *
 */
public class ModuleTypeDTO {
	
	// name of the type 
	public final String type;
	
	// category for grouping module types in a UI builder
	public final String category;
	
	// properties that should be set when creating an instance of this Module type
	public final ModulePropertyDTO[] properties;
	
	// whether this type can be trained or not
	public final boolean trainable;
	
	public ModuleTypeDTO(String type, String category, 
			boolean trainable, ModulePropertyDTO... properties){
		this.type = type;
		this.category = category;
		this.trainable = trainable;
		this.properties = properties;
	}
}
