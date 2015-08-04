package be.iminds.iot.dianne.api.nn.module.dto;

import java.util.List;
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
	public UUID nnId;
	
	// name of the neural network
	// can be used to fetch the NeuralNetworkDTO from DianneRepository
	public String name;
	
	// The list of ModuleInstances that this neural network instance is composed of
	public List<ModuleInstanceDTO> modules;
}
