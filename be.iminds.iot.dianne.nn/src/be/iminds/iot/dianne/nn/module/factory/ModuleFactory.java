package be.iminds.iot.dianne.nn.module.factory;

import java.util.Dictionary;
import java.util.List;

import be.iminds.iot.dianne.nn.module.Module;
import be.iminds.iot.dianne.nn.module.description.ModuleType;
import be.iminds.iot.dianne.tensor.TensorFactory;

public interface ModuleFactory {

	Module createModule(TensorFactory factory, Dictionary<String, ?> config) throws InstantiationException;
	
	List<ModuleType> getAvailableModuleTypes();
	
	ModuleType getModuleType(String name);
}
