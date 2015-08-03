package be.iminds.iot.dianne.api.nn.module.dto;

/**
 * A ModuleType declares how a certain module should look like.
 * 
 * The type String is used to identify the ModuleType and is used in the ModuleDTO
 * to determine the type. This DTO can be used for a UI builder to discover the properties
 * that should be set to create an instance of such a Module type.
 * 
 * @author tverbele
 *
 */
public class ModuleTypeDTO {
	
	// name of the type 
	public String type;
	
	// category for grouping module types in a UI builder
	public String category;
	
	// properties that should be set when creating an instance of this Module type
	public ModulePropertyDTO[] properties;
	
	// whether this type can be trained or not
	public boolean trainable;
	
}
