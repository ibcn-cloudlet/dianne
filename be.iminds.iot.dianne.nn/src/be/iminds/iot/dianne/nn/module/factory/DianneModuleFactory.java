package be.iminds.iot.dianne.nn.module.factory;

import java.util.Dictionary;

import be.iminds.iot.dianne.nn.module.Module;
import be.iminds.iot.dianne.nn.module.activation.Sigmoid;
import be.iminds.iot.dianne.nn.module.activation.Tanh;
import be.iminds.iot.dianne.nn.module.io.InputImpl;
import be.iminds.iot.dianne.nn.module.io.OutputImpl;
import be.iminds.iot.dianne.nn.module.layer.Linear;
import be.iminds.iot.dianne.tensor.TensorFactory;

public class DianneModuleFactory implements ModuleFactory {

	private TensorFactory factory;
	
	@Override
	public Module createModule(Dictionary<String, ?> config)
			throws InstantiationException {
		Module module = null;
		
		// TODO have a better design for this?
		// for now just hard code an if/else for each known module
		String type = (String)config.get("module.type");
		
		if(type.equals("Linear")){
			int input = Integer.parseInt((String)config.get("module.linear.input"));
			int output = Integer.parseInt((String)config.get("module.linear.output"));
			
			module = new Linear(factory, input, output);
		} else if(type.equals("Tanh")){
			module = new Tanh(factory); 
		} else if(type.equals("Sigmoid")){
			module = new Sigmoid(factory);
		} else if(type.equals("Input")){
			module = new InputImpl(factory); 
		} else if(type.equals("Output")){
			module = new OutputImpl(factory);
		} else if(type.equals("anotherTypeHere")){
			// instantiate other types ... 
		}
		
		if(module==null){
			throw new InstantiationException("Could not instantiate module of type "+type);
		}
		
		return module;
	}

}
