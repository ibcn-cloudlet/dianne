package be.iminds.iot.dianne.api.nn.platform;

import java.util.List;
import java.util.Map;
import java.util.UUID;

import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkInstanceDTO;

/**
 * The NeuralNetworkManager is a high-level management entity to control the deployment
 * of (distributed) neural network instances.
 * 
 * @author tverbele
 *
 */
public interface NeuralNetworkManager {

	/**
	 * Deploy an instance of a neural network on a given runtime
	 * 
	 * @param name name of the neural network
	 * @param runtimeId identifier of the Dianne runtime to deploy the neural network modules on
	 * @return NeuralNetworkInstanceDTO of the deployed neural network
	 * @throws InstantiationException thrown when failed to deploy all neural network modules
	 */
	NeuralNetworkInstanceDTO deployNeuralNetwork(String name, UUID runtimeId) throws InstantiationException;
	
	/**
	 * Deploy an instance of a neural network on a given set of runtimes
	 * 
	 * @param name name of the neural network
	 * @param runtimeId identifier of the Dianne runtime to deploy the neural network modules on
	 * @param deployment a map mapping moduleIds to runtimeIds representing the requested deployment; moduleIds not mentioned in the map are deployed to runtimeId
	 * @return NeuralNetworkInstanceDTO of the deployed neural network
	 * @throws InstantiationException thrown when failed to deploy all neural network modules
	 */
	NeuralNetworkInstanceDTO deployNeuralNetwork(String name, UUID runtimeId, Map<UUID, UUID> deployment) throws InstantiationException;

	/**
	 * Undeploy a neural network instance
	 * 
	 * @param nn the neural network instance to undeploy
	 */
	void undeployNeuralNetwork(NeuralNetworkInstanceDTO nn);
	
	/**
	 * Get a list of deployed neural networks
	 * 
	 * @return list of deployed neural networks
	 */
	List<NeuralNetworkInstanceDTO> getNeuralNetworks();
	
	/**
	 * Get a list of known neural networks
	 * 
	 * This is the aggregated list of all neural networks available 
	 * in the DianneRepositories this NeuralNetworkManager has access to.
	 * 
	 * @return the list of names that this NeuralNetworkManager can deploy
	 */
	List<String> getSupportedNeuralNetworks();
	
	/**
	 * A list of available Dianne runtimes this NeuralNetworkManager can deploy modules to
	 * 
	 * @return list of available Dianne runtimes
	 */
	List<UUID> getRuntimes();

}
