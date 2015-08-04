package be.iminds.iot.dianne.api.nn.runtime;

import java.util.List;
import java.util.UUID;

import be.iminds.iot.dianne.api.nn.module.dto.ModuleDTO;
import be.iminds.iot.dianne.api.nn.module.dto.ModuleInstanceDTO;
import be.iminds.iot.dianne.api.nn.module.dto.ModuleTypeDTO;

public interface ModuleManager {

	ModuleInstanceDTO deployModule(ModuleDTO dto, UUID nnId) throws InstantiationException;
	
	void undeployModule(ModuleInstanceDTO module);
	
	List<ModuleInstanceDTO> getModules();
	
	List<ModuleTypeDTO> getSupportedModules();
	
}
