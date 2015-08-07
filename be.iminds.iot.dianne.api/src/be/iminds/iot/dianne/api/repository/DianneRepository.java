package be.iminds.iot.dianne.api.repository;

import java.io.IOException;
import java.util.List;
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
	NeuralNetworkDTO loadNeuralNetwork(String nnName) throws IOException;
	
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
	Tensor loadParameters(UUID moduleId, String... tag) throws IOException;
	
	/**
	 * Store parameters for a given moduleId
	 * 
	 * @param parameters the parameters Tensor
	 * @param moduleId the moduleId for which these parameters are applicable
	 * @param tag optional tags for the parameters
	 */
	void storeParameters(Tensor parameters, UUID moduleId, String... tag);
	
	/**
	 * Update the parameters for a given moduleId with this diff
	 * 
	 * @param accParameters a diff with the old parameters
	 * @param moduleId the moduleId for which these parameters are applicable
	 * @param tag optional tags for the parameters
	 */
	void accParameters(Tensor accParameters, UUID moduleId, String... tag);
	

	// these are some helper methods for saving the jsplumb layout of the UI builder
	// of utterly no importance for the rest and can be ignored...
	String loadLayout(String nnName) throws IOException;
	
	void storeLayout(String nnName, String layout);
}
