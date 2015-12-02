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
package be.iminds.iot.dianne.api.nn.runtime;

import java.util.List;
import java.util.UUID;

import be.iminds.iot.dianne.api.nn.module.dto.ModuleDTO;
import be.iminds.iot.dianne.api.nn.module.dto.ModuleInstanceDTO;
import be.iminds.iot.dianne.api.nn.module.dto.ModuleTypeDTO;
import be.iminds.iot.dianne.tensor.Tensor;

/**
 * The DianneRuntime is responsible for the correct deployment and configuration 
 * of individual Modules. It will call the right factory for creating module instances,
 * and correctly configure the next and previous of each Module.
 * 
 * @author tverbele
 *
 */
public interface DianneRuntime {

	/**
	 * The unique identifier of this runtime. Is usually the same as the OSGi framework UUID
	 * 
	 * @return the uuid of this DianneRuntime
	 */
	UUID getRuntimeId();
	
	/**
	 * A human readable name for this runtime (used in UI / CLI)
	 * @return
	 */
	String getRuntimeName();
	
	/**
	 * Deploy a single Module on this runtime
	 * 
	 * @param dto the ModuleDTO describing which Module to deploy
	 * @param nnId the neural network instance this module instance will belong to
	 * @param tags the tags of parameters to load for this module
	 * @return the ModuleInstanceDTO of the deployed module
	 * @throws InstantiationException when it failed to construct the desired module instance
	 */
	ModuleInstanceDTO deployModule(ModuleDTO dto, UUID nnId, String... tags);
	
	/**
	 * Deploy a single Module on this runtime and provide parameters
	 * 
	 * @param dto the ModuleDTO describing which Module to deploy
	 * @param nnId the neural network instance this module instance will belong to
	 * @param parameters the parameters to initialize this module with
	 * @return the ModuleInstanceDTO of the deployed module
	 * @throws InstantiationException when it failed to construct the desired module instance
	 */
	ModuleInstanceDTO deployModule(ModuleDTO dto, UUID nnId, Tensor parameters);
	
	/**
	 * Undeploy a single ModuleInstance on this runtime
	 * 
	 * @param module the module instance to undeploy
	 */
	void undeployModule(ModuleInstanceDTO module);
	
	/**
	 * Undeploy all ModuleInstances from a given neural network instance that reside on this runtime
	 * 
	 * @param nnId the neural network instance id of the neural network instance to undeploy
	 */
	void undeployModules(UUID nnId);
	
	/**
	 * Get a list of all ModuleInstances that are deployed on this runtime
	 * 
	 * @return list of deployed module instances
	 */
	List<ModuleInstanceDTO> getModules();
	
	/**
	 * Get the parameters of a currently deployed module
	 * @param module
	 * @return
	 */
	Tensor getModuleParameters(ModuleInstanceDTO module);
	
	/**
	 * Get a list of supported module types that this runtime can deploy. 
	 * 
	 * This is the aggregated list of supported ModuleTypes that the ModuleFactories support
	 * that this runtime as access to.
	 * 
	 * @return list of supported module types
	 */
	List<ModuleTypeDTO> getSupportedModules();
	
}
