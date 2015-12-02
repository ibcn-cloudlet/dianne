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
package be.iminds.iot.dianne.api.nn.module;

import java.util.UUID;

import be.iminds.iot.dianne.tensor.Tensor;

/**
 * Called when a Module has done a backward pass. 
 * 
 * This interface can be registered as an OSGi service with the service property
 *  targets = String[]{nnId:moduleId, nnId:moduleId2}
 * in order to listen to only specific modules from certain neural network instances.
 * 
 * In case only a nnId is given as targets property, this backward listener will only 
 * listen to the Input modules of this neural network instance.
 *   
 * @author tverbele
 *
 */
public interface BackwardListener {

	/**
	 * Called when a Module has performed a backward pass
	 * @param moduleId the moduleId of the module whos backward was called
	 * @param output a copy of the gradInput data
	 * @param tags a copy of the tags provided
	 */
	void onBackward(final UUID moduleId, final Tensor gradInput, final String... tags);
	
	default void onError(final UUID moduleId, final ModuleException e, final String... tags){
		e.printStackTrace();
	}
}
