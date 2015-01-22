package be.iminds.iot.dianne.nn.module.factory;

import java.util.Dictionary;

import be.iminds.iot.dianne.nn.module.Module;

public interface ModuleFactory {

	Module createModule(Dictionary<String, ?> config) throws InstantiationException;
	
}
