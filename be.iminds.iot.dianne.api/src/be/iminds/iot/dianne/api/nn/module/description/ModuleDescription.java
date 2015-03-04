package be.iminds.iot.dianne.api.nn.module.description;

import java.util.Map;
import java.util.UUID;

public class ModuleDescription {

	private final UUID id;
	private final ModuleType type;
	private final Map<ModuleProperty, Object> parameters;
	
	public ModuleDescription(UUID id, ModuleType type, Map<ModuleProperty, Object> parameters){
		this.id = id;
		this.type = type;
		this.parameters = parameters;
	}
	
	public UUID getId(){
		return id;
	}
	
	public ModuleType getType(){
		return type;
	}
	
	public Object getParameter(String id){
		// ModuleProperty is hashed by its id
		return parameters.get(id);
	}
}
