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
	public String name;
	
	// the ModuleDTOs that this neural network consists of
	public List<ModuleDTO> modules;
	
}
