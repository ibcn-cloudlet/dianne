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
package be.iminds.iot.dianne.api.nn.platform;

import java.util.List;
import java.util.Map;
import java.util.UUID;

import be.iminds.iot.dianne.api.nn.module.dto.ModuleDTO;
import be.iminds.iot.dianne.api.nn.module.dto.ModuleInstanceDTO;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkDTO;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkInstanceDTO;

/**
 * The DiannePlatform interface is the main entry point to control the deployment
 * of (distributed) neural network instances.
 * 
 * @author tverbele
 *
 */
public interface DiannePlatform {

	/**
	 * Deploy an instance of a neural network on the local runtime
	 * @param name name of the neural network 
	 * @param tags tags of the weights to load
	 * @return NeuralNetworkInstanceDTO of the deployed neural network
	 * @throws InstantiationException
	 */
	NeuralNetworkInstanceDTO deployNeuralNetwork(String name, String... tags) throws InstantiationException;
	NeuralNetworkInstanceDTO deployNeuralNetwork(String name, Map<String, String> properties, String... tags) throws InstantiationException;
	NeuralNetworkInstanceDTO deployNeuralNetwork(String name, String description, String... tags) throws InstantiationException;
	NeuralNetworkInstanceDTO deployNeuralNetwork(String name, String description, Map<String, String> properties, String... tags) throws InstantiationException;

	NeuralNetworkInstanceDTO deployNeuralNetwork(NeuralNetworkDTO nn, String... tags) throws InstantiationException;
	NeuralNetworkInstanceDTO deployNeuralNetwork(NeuralNetworkDTO nn, String description, String... tags) throws InstantiationException;


	
	/**
	 * Deploy an instance of a neural network on a given runtime
	 * 
	 * @param name name of the neural network
	 * @param runtimeId identifier of the Dianne runtime to deploy the neural network modules on
	 * @param tags tags of the weights to load
	 * @return NeuralNetworkInstanceDTO of the deployed neural network
	 * @throws InstantiationException thrown when failed to deploy all neural network modules
	 */
	NeuralNetworkInstanceDTO deployNeuralNetwork(String name, UUID runtimeId, String... tags) throws InstantiationException;
	NeuralNetworkInstanceDTO deployNeuralNetwork(String name, Map<String, String> properties, UUID runtimeId, String... tags) throws InstantiationException;
	NeuralNetworkInstanceDTO deployNeuralNetwork(String name, String description, UUID runtimeId, String... tags) throws InstantiationException;
	NeuralNetworkInstanceDTO deployNeuralNetwork(String name, String description, Map<String, String> properties, UUID runtimeId, String... tags) throws InstantiationException;
	
	NeuralNetworkInstanceDTO deployNeuralNetwork(NeuralNetworkDTO nn, UUID runtimeId, String... tags) throws InstantiationException;
	NeuralNetworkInstanceDTO deployNeuralNetwork(NeuralNetworkDTO nn, String description, UUID runtimeId, String... tags) throws InstantiationException;

	
	
	/**
	 * Deploy an instance of a neural network on a given set of runtimes
	 * 
	 * @param name name of the neural network
	 * @param runtimeId identifier of the Dianne runtime to deploy the neural network modules on
	 * @param deployment a map mapping moduleIds to runtimeIds representing the requested deployment; moduleIds not mentioned in the map are deployed to runtimeId
	 * @param tags tags of the weights to load
	 * @return NeuralNetworkInstanceDTO of the deployed neural network
	 * @throws InstantiationException thrown when failed to deploy all neural network modules
	 */
	NeuralNetworkInstanceDTO deployNeuralNetwork(String name, UUID runtimeId, Map<UUID, UUID> deployment, String... tags) throws InstantiationException;
	NeuralNetworkInstanceDTO deployNeuralNetwork(String name, Map<String, String> properties, UUID runtimeId, Map<UUID, UUID> deployment, String... tags) throws InstantiationException;
	NeuralNetworkInstanceDTO deployNeuralNetwork(String name, String description, UUID runtimeId, Map<UUID, UUID> deployment, String... tags) throws InstantiationException;
	NeuralNetworkInstanceDTO deployNeuralNetwork(String name, String description, Map<String, String> properties, UUID runtimeId, Map<UUID, UUID> deployment, String... tags) throws InstantiationException;

	NeuralNetworkInstanceDTO deployNeuralNetwork(NeuralNetworkDTO nn, UUID runtimeId, Map<UUID, UUID> deployment, String... tags) throws InstantiationException;
	NeuralNetworkInstanceDTO deployNeuralNetwork(NeuralNetworkDTO nn, String description, UUID runtimeId, Map<UUID, UUID> deployment, String... tags) throws InstantiationException;


	/**
	 * Undeploy a neural network instance
	 * 
	 * @param nn the neural network instance to undeploy
	 */
	void undeployNeuralNetwork(NeuralNetworkInstanceDTO nn);
	void undeployNeuralNetwork(UUID nnId);
	
	
	/**
	 * Get a list of deployed neural networks
	 * 
	 * @return list of deployed neural networks
	 */
	List<NeuralNetworkInstanceDTO> getNeuralNetworkInstances();
	
	/**
	 * Get a neural network instance DTO by its instance id
	 * @param nnId
	 * @return
	 */
	NeuralNetworkInstanceDTO getNeuralNetworkInstance(UUID nnId);

	
	/**
	 * Deploy instances of neural network modules on a given runtime. If a nnId is given and some of the modules are already
	 * deployed elsewhere, these modules will be migrated.
	 * 
	 * @param nnId the id of the neural network under which these should be deployed - if null a new nnId will be generated
	 * @param modules the modules to deploy
	 * @param runtimeId identifier of the Dianne runtime to deploy the neural network modules on
	 * @param tags tags of the weights to load
	 * @return ModuleInstanceDTOs of each deployed module
	 * @throws InstantiationException thrown when failed to deploy all neural network modules
	 */
	List<ModuleInstanceDTO> deployModules(UUID nnId, List<ModuleDTO> modules, UUID runtimeId, String... tags) throws InstantiationException;
	List<ModuleInstanceDTO> deployModules(UUID nnId, String name, List<ModuleDTO> modules, UUID runtimeId, String... tags) throws InstantiationException;
	List<ModuleInstanceDTO> deployModules(UUID nnId, String name, String description, List<ModuleDTO> modules, UUID runtimeId, String... tags) throws InstantiationException;

	
	/**
	 * Undeploy module instances
	 * 
	 * @param moduleInstances
	 */
	void undeployModules(List<ModuleInstanceDTO> moduleInstances);
	
	
	
	/**
	 * Get a list of available neural networks
	 * 
	 * This is the aggregated list of all neural networks available 
	 * in the DianneRepositories.
	 * 
	 * @return the list of names that can be deployed
	 */
	List<String> getAvailableNeuralNetworks();
	
	/**
	 * Get the structure of a neural network by name
	 * @param name
	 * @return the NeuralNetworkDTO describing the neural network structure
	 */
	NeuralNetworkDTO getAvailableNeuralNetwork(String name);
	
	
	/**
	 * A map of available Dianne runtimes to which modules can be deployed to
	 * 
	 * @return list of available Dianne runtimes
	 */
	Map<UUID, String> getRuntimes();

}
