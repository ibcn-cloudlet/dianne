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
package be.iminds.iot.dianne.api.repository;

import java.io.IOException;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;

import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkDTO;
import be.iminds.iot.dianne.tensor.Tensor;

/**
 * The DianneRepository offers access to known neural networks and their stored parameters.
 * 
 * @author tverbele
 *
 */
public interface DianneRepository {

	/**
	 * Get a list of available neural networks
	 * @return list of available neural networks
	 */
	List<String> availableNeuralNetworks();
	
	/**
	 * Get a detailed description of a neural network
	 * @param nnName the name of the neural network
	 * @return the NeuralNetworkDTO representing this neural network
	 * @throws IOException 
	 */
	NeuralNetworkDTO loadNeuralNetwork(String nnName) ;
	
	/**
	 * Store a new neural network
	 * @param nn the NeuralNetworkDTO representing the neural network
	 */
	void storeNeuralNetwork(NeuralNetworkDTO nn);
	
	/**
	 * Load the parameters for a given moduleId, optionally with some tags
	 * 
	 * @param moduleId moduleId for which the parameters to load
	 * @param tag optional tags for the parameters
	 * @return the parameter Tensor
	 * @throws IOException
	 */
	Tensor loadParameters(UUID moduleId, String... tag) throws Exception;
	
	/**
	 * Load the parameters for a number of moduleIds, optionally with some tags
	 * 
	 * @param moduleIds moduleIdds for which the parameters to load
	 * @param tag optional tags for the parameters
	 * @return the parameters Tensors mapped by moduleId
	 * @throws IOException
	 */
	Map<UUID, Tensor> loadParameters(Collection<UUID> moduleIds, String... tag) throws Exception;
	
	/**
	 * Load all parameters for a given neural network name for some tags
	 * 
	 * @param nnName name of the neural network
	 * @param tag optional tags for the parameters
	 * @return the parameters Tensor mapped by moduleId
	 */
	Map<UUID, Tensor> loadParameters(String nnName, String... tag) throws Exception;
	
	/**
	 * Store parameters for a given moduleId
	 *
	 * @param nnId the nn instance these parameters originate from
	 * @param moduleId the moduleId for which these parameters are applicable
	 * @param parameters the parameters Tensor
	 * @param tag optional tags for the parameters
	 */
	void storeParameters(UUID nnId, UUID moduleId, Tensor parameters, String... tag);
	
	/**
	 * List all available tags for a given module
	 * @param moduleId
	 * @return list of tags available in the repository
	 */
	Set<String> listTags(UUID moduleId);
	
	/**
	 * List all available tags for a given neural network
	 * @param nnName
	 * @return list of tags available in the repository
	 */
	Set<String> listTags(String nnName);
	
	/**
	 * Update the parameters for a given moduleId with this diff
	 * 
	 * @param nnId the nn instance these parameters originate from
	 * @param moduleId the moduleId for which these parameters are applicable
	 * @param accParameters a diff with the old parameters
	 * @param tag optional tags for the parameters
	 */
	void accParameters(UUID nnId, UUID moduleId, Tensor accParameters, String... tag);
	
	/**
	 * Store parameters for a number of modules
	 *
	 * @param nnId the nn instance these parameters originate from
	 * @param parameters the parameters Tensors mapped by moduleIds
	 * @param tag optional tags for the parameters
	 */
	void storeParameters(UUID nnId, Map<UUID, Tensor> parameters, String... tag);
	 
	/**
	 * Update the parameters for a number of modules with this diff
	 * 
	 * @param nnId the nn instance these parameters originate from
	 * @param accParameters a diff with the old parameters mapped by moduleId
	 * @param tag optional tags for the parameters
	 */
	void accParameters(UUID nnId, Map<UUID, Tensor> accParameters, String... tag);
	
	/**
	 * Check how many space is left on this device to store things
	 * @return
	 */
	long spaceLeft();

	// these are some helper methods for saving the jsplumb layout of the UI builder
	// of utterly no importance for the rest and can be ignored...
	String loadLayout(String nnName) throws IOException;
	
	void storeLayout(String nnName, String layout);
	

}
