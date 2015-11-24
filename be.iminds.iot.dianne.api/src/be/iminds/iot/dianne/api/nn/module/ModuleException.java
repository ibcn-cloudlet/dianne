package be.iminds.iot.dianne.api.nn.module;

import java.util.UUID;

public class ModuleException extends Exception {

	public UUID moduleId;
	public String type;
	
	public ModuleException(UUID moduleId, String type, boolean forward, Exception cause){
		super("Error in "+(forward ? "forward" : "backward")+" of module "+type+" "+moduleId+": "+cause.getMessage());
		this.moduleId = moduleId;
		this.type = type;
	}
}
