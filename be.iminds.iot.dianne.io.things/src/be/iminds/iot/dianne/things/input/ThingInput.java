package be.iminds.iot.dianne.things.input;

import java.util.List;
import java.util.UUID;
import java.util.concurrent.CopyOnWriteArrayList;

import org.osgi.framework.BundleContext;

import be.iminds.iot.dianne.api.io.InputDescription;
import be.iminds.iot.dianne.api.nn.module.Input;

public abstract class ThingInput {

	protected final UUID id;
	protected final String name;
	protected String type;
	
	protected final List<Input> inputs = new CopyOnWriteArrayList<>();
	
	public ThingInput(UUID id, String name, String type){
		this.id = id;
		this.name = name;
		this.type = type;
	}
	
	public InputDescription getInputDescription(){
		return new InputDescription(name, type);
	}
	
	public void connect(Input input, BundleContext context){
		inputs.add(input);
	}
	
	public void disconnect(Input input){
		inputs.remove(input);
	}
	
}
