package be.iminds.iot.dianne.api.nn.platform;

import java.util.List;
import java.util.Map;
import java.util.UUID;

import be.iminds.iot.dianne.api.nn.module.dto.ModuleDTO;
import be.iminds.iot.dianne.api.nn.module.dto.ModuleInstanceDTO;
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
	 * Deploy an instance of a neural network on the local runtime
	 * @param name name of the neural network 
	 * @return NeuralNetworkInstanceDTO of the deployed neural network
	 * @throws InstantiationException
	 */
	NeuralNetworkInstanceDTO deployNeuralNetwork(String name) throws InstantiationException;
	
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
	List<NeuralNetworkInstanceDTO> getNeuralNetworkInstances();
	
	/**
	 * Get a neural network instance DTO by its instance id
	 * @param nnId
	 * @return
	 */
	NeuralNetworkInstanceDTO getNeuralNetworkInstance(UUID nnId);

	/**
	 * Get an NeuralNetwork interface for synchronously forward and backward tensors
	 * through the network
	 */
	NeuralNetwork getNeuralNetwork(UUID nnId);
	
	/**
	 * Deploy instances of neural network modules on a given runtime. If a nnId is given and some of the modules are already
	 * deployed elsewhere, these modules will be migrated.
	 * 
	 * @param nnId the id of the neural network under which these should be deployed - if null a new nnId will be generated
	 * @param modules the modules to deploy
	 * @param runtimeId identifier of the Dianne runtime to deploy the neural network modules on
	 * @return ModuleInstanceDTOs of each deployed module
	 * @throws InstantiationException thrown when failed to deploy all neural network modules
	 */
	List<ModuleInstanceDTO> deployModules(UUID nnId, List<ModuleDTO> modules, UUID runtimeId) throws InstantiationException;
	
	/**
	 * Undeploy module instances
	 * 
	 * @param moduleInstances
	 */
	void undeployModules(List<ModuleInstanceDTO> moduleInstances);
	
	
	
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
