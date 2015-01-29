package be.iminds.iot.dianne.nn.module.description;

import java.util.Collections;
import java.util.List;

public class ModuleDescription {

	private final String name;
	private final List<ModuleProperty> properties;
	
	public ModuleDescription(String name, List<ModuleProperty> properties){
		this.name = name;
		this.properties = Collections.unmodifiableList(properties);
	}

	public String getName() {
		return name;
	}

	public List<ModuleProperty> getProperties() {
		return properties;
	}
	
}
