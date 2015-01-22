package be.iminds.iot.dianne.nn.module.factory;

import java.util.Dictionary;
import java.util.UUID;

import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;

import be.iminds.iot.dianne.nn.module.Module;
import be.iminds.iot.dianne.nn.module.activation.Sigmoid;
import be.iminds.iot.dianne.nn.module.activation.Tanh;
import be.iminds.iot.dianne.nn.module.io.InputImpl;
import be.iminds.iot.dianne.nn.module.io.OutputImpl;
import be.iminds.iot.dianne.nn.module.layer.Linear;
import be.iminds.iot.dianne.tensor.TensorFactory;

@Component
public class DianneModuleFactory implements ModuleFactory {

	@Override
	public Module createModule(TensorFactory factory, Dictionary<String, ?> config)
			throws InstantiationException {
		Module module = null;
		
		// TODO have a better design for this?
		// for now just hard code an if/else for each known module
		String type = (String)config.get("module.type");
		UUID id = UUID.fromString((String)config.get("module.id"));
		
		if(type.equals("Linear")){
			int input = Integer.parseInt((String)config.get("module.linear.input"));
			int output = Integer.parseInt((String)config.get("module.linear.output"));
			
			module = new Linear(factory, id, input, output);
		} else if(type.equals("Tanh")){
			module = new Tanh(factory, id); 
		} else if(type.equals("Sigmoid")){
			module = new Sigmoid(factory, id);
		} else if(type.equals("Input")){
			module = new InputImpl(factory, id); 
		} else if(type.equals("Output")){
			module = new OutputImpl(factory, id);
		} else if(type.equals("anotherTypeHere")){
			// instantiate other types ... 
		}
		
		if(module==null){
			throw new InstantiationException("Could not instantiate module of type "+type);
		}
		
		return module;
	}

}
