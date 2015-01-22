package be.iminds.iot.dianne.nn.module;

import java.util.Dictionary;

public interface ModuleFactory {

	Module createModule(Dictionary<String, ?> config) throws InstantiationException;
	
}
