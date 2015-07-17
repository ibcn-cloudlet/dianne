package be.iminds.iot.dianne.api.nn.runtime;

import java.util.Dictionary;
import java.util.List;
import java.util.UUID;

import be.iminds.iot.dianne.api.nn.module.description.ModuleType;

public interface ModuleManager {

	public UUID deployModule(Dictionary<String, ?> properties) throws InstantiationException;
	
	public void undeployModule(UUID moduleId);
	
	public List<UUID> getModules();
	
	public List<ModuleType> getSupportedModules();
	
}
