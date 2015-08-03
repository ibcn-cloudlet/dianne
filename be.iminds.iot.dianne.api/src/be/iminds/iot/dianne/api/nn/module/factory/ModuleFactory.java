package be.iminds.iot.dianne.api.nn.module.factory;

import java.util.List;

import be.iminds.iot.dianne.api.nn.module.Module;
import be.iminds.iot.dianne.api.nn.module.dto.ModuleDTO;
import be.iminds.iot.dianne.api.nn.module.dto.ModuleTypeDTO;

public interface ModuleFactory {

	Module createModule(ModuleDTO dto) throws InstantiationException;
	
	List<ModuleTypeDTO> getAvailableModuleTypes();
	
	ModuleTypeDTO getModuleType(String name);
}
