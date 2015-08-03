package be.iminds.iot.dianne.api.nn.module.dto;

import java.util.Map;
import java.util.UUID;

/**
 * Represents a single Module of a Neural Network.
 * 
 * @author tverbele
 *
 */
public class ModuleDTO {

	// UUID of this Module
	public UUID id;
	
	// Type of this Module
	//  this maps to a ModuleTypeDTO, is used by a factory 
	//  to create an instance of this Module
	public String type;
	
	// UUID(s) of the next Modules in the neural network
	public UUID[] next;
	// UUID(s) of the previous Modules in the neural network
	public UUID[] prev;
	
	// Specific properties for this Module
	public Map<String, String> properties;

}
