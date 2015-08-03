package be.iminds.iot.dianne.api.nn.module.dto;

/**
 * A configuration property for a Module.
 * 
 * Consists of a human-readable name, and id used as property key,
 * and the name of the expected class type.
 * 
 * @author tverbele
 *
 */
public class ModulePropertyDTO {

	// human-readable name for this property
	public String name;
	
	// id to be used as property key in an actual configuration
	public String id;
	
	// clazz that is expected as property value
	public String clazz;
}
