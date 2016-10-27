package be.iminds.iot.dianne.things.input;

import java.util.UUID;

import org.osgi.framework.BundleContext;

import be.iminds.iot.dianne.api.io.InputDescription;
import be.iminds.iot.dianne.api.nn.module.Input;

public abstract class ThingInput {

	public final UUID id;
	public final String name;
	public final String type;
	
	public ThingInput(UUID id, String name, String type){
		this.id = id;
		this.name = name;
		this.type = type;
	}
	
	public InputDescription getInputDescription(){
		return new InputDescription(name, type);
	}
	
	public abstract void connect(Input input, BundleContext context);
	
	public abstract void disconnect();
	
}
