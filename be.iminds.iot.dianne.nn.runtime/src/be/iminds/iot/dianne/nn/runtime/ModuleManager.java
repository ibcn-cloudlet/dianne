package be.iminds.iot.dianne.nn.runtime;

import java.util.Dictionary;
import java.util.List;
import java.util.UUID;

import be.iminds.iot.dianne.nn.module.description.ModuleDescription;

public interface ModuleManager {

	public UUID deployModule(Dictionary<String, ?> properties) throws InstantiationException;
	
	public void undeployModule(UUID moduleId);
	
	public List<ModuleDescription> getSupportedModules();
	
	// TODO list deployed modules?
}
