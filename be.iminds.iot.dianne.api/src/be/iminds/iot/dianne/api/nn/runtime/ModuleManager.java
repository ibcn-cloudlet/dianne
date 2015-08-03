package be.iminds.iot.dianne.api.nn.runtime;

import java.util.List;
import java.util.UUID;

import be.iminds.iot.dianne.api.nn.module.dto.ModuleDTO;
import be.iminds.iot.dianne.api.nn.module.dto.ModuleTypeDTO;

public interface ModuleManager {

	public UUID deployModule(ModuleDTO dto) throws InstantiationException;
	
	public void undeployModule(UUID moduleId);
	
	public List<UUID> getModules();
	
	public List<ModuleTypeDTO> getSupportedModules();
	
}
