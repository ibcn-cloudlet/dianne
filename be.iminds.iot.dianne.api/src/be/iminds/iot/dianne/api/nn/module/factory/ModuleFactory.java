package be.iminds.iot.dianne.api.nn.module.factory;

import java.util.List;

import be.iminds.iot.dianne.api.nn.module.Module;
import be.iminds.iot.dianne.api.nn.module.dto.ModuleDTO;
import be.iminds.iot.dianne.api.nn.module.dto.ModuleTypeDTO;

/**
 * A ModuleFactory knows how to create Module instances from a ModuleDTO.
 * 
 * The factory also announces all ModuleTypes it knows to construct.
 * @author tverbele
 *
 */
public interface ModuleFactory {

	/**
	 * Create a new Module instance from a ModuleDTO
	 * @param dto the dto describing the module to construct
	 * @return a reference to the constructed Module
	 * @throws InstantiationException thrown when it failed to instantiate the Module
	 */
	Module createModule(ModuleDTO dto) throws InstantiationException;
	
	/**
	 * Get the list of module types that this factory can construct
	 * 
	 * @return available module types
	 */
	List<ModuleTypeDTO> getAvailableModuleTypes();
	
	/**
	 * Get a detailed ModuleTypeDTO for a given module type name
	 * 
	 * @param name name of the module type
	 * @return the detailed ModuleDTO matching this type, or null if this type is not available
	 */
	ModuleTypeDTO getModuleType(String name);
}
