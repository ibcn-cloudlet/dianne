package be.iminds.iot.dianne.api.nn.runtime;

import java.util.List;
import java.util.UUID;

import be.iminds.iot.dianne.api.nn.module.dto.ModuleDTO;
import be.iminds.iot.dianne.api.nn.module.dto.ModuleInstanceDTO;
import be.iminds.iot.dianne.api.nn.module.dto.ModuleTypeDTO;

public interface ModuleManager {

	public ModuleInstanceDTO deployModule(ModuleDTO dto, UUID nnId) throws InstantiationException;
	
	public void undeployModule(ModuleInstanceDTO module);
	
	public List<ModuleInstanceDTO> getModules();
	
	public List<ModuleTypeDTO> getSupportedModules();
	
}
