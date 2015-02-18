package be.iminds.iot.dianne.nn.module.factory;

import java.util.ArrayList;
import java.util.Dictionary;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Component;

import be.iminds.iot.dianne.nn.module.AbstractModule;
import be.iminds.iot.dianne.nn.module.Module;
import be.iminds.iot.dianne.nn.module.activation.ReLU;
import be.iminds.iot.dianne.nn.module.activation.Sigmoid;
import be.iminds.iot.dianne.nn.module.activation.Softmax;
import be.iminds.iot.dianne.nn.module.activation.Tanh;
import be.iminds.iot.dianne.nn.module.activation.Threshold;
import be.iminds.iot.dianne.nn.module.conv.SpatialConvolution;
import be.iminds.iot.dianne.nn.module.conv.SpatialMaxPooling;
import be.iminds.iot.dianne.nn.module.description.ModuleDescription;
import be.iminds.iot.dianne.nn.module.description.ModuleProperty;
import be.iminds.iot.dianne.nn.module.fork.Duplicate;
import be.iminds.iot.dianne.nn.module.fork.Split;
import be.iminds.iot.dianne.nn.module.io.InputImpl;
import be.iminds.iot.dianne.nn.module.io.OutputImpl;
import be.iminds.iot.dianne.nn.module.join.Accumulate;
import be.iminds.iot.dianne.nn.module.join.Concat;
import be.iminds.iot.dianne.nn.module.layer.Linear;
import be.iminds.iot.dianne.nn.module.preprocessing.Normalization;
import be.iminds.iot.dianne.tensor.TensorFactory;

@Component(property={"aiolos.export=false"})
public class DianneModuleFactory implements ModuleFactory {

	private ExecutorService executor = Executors.newCachedThreadPool();
	
	private final Map<String, ModuleDescription> supportedModules = new HashMap<String, ModuleDescription>();
	
	@Activate
	public void activate(){
		// build list of supported modules
		// TODO use reflection for this?
		
		{
			List<ModuleProperty> properties = new ArrayList<ModuleProperty>();
			properties.add(new ModuleProperty("Input size", "input"));
			properties.add(new ModuleProperty("Output size", "output"));
			ModuleDescription description = new ModuleDescription("Linear", "Linear", properties);
			supportedModules.put(description.getType(), description);
		}
		{
			List<ModuleProperty> properties = new ArrayList<ModuleProperty>();
			ModuleDescription description = new ModuleDescription("Sigmoid", "Activation", properties);
			supportedModules.put(description.getType(), description);
		}
		{
			List<ModuleProperty> properties = new ArrayList<ModuleProperty>();
			ModuleDescription description = new ModuleDescription("Tanh", "Activation", properties);
			supportedModules.put(description.getType(), description);
		}
		{
			List<ModuleProperty> properties = new ArrayList<ModuleProperty>();
			ModuleDescription description = new ModuleDescription("Softmax", "Activation", properties);
			supportedModules.put(description.getType(), description);
		}		
		{
			List<ModuleProperty> properties = new ArrayList<ModuleProperty>();
			ModuleDescription description = new ModuleDescription("ReLU", "Activation", properties);
			supportedModules.put(description.getType(), description);
		}
		{
			List<ModuleProperty> properties = new ArrayList<ModuleProperty>();
			properties.add(new ModuleProperty("Threshold value", "thresh"));
			properties.add(new ModuleProperty("Replacement value", "val"));
			ModuleDescription description = new ModuleDescription("Threshold", "Activation", properties);
			supportedModules.put(description.getType(), description);
		}
		{
			List<ModuleProperty> properties = new ArrayList<ModuleProperty>();
			ModuleDescription description = new ModuleDescription("Input", "Input", properties);
			supportedModules.put(description.getType(), description);
		}
		{
			List<ModuleProperty> properties = new ArrayList<ModuleProperty>();
			ModuleDescription description = new ModuleDescription("Output", "Output", properties);
			supportedModules.put(description.getType(), description);
		}
		{
			List<ModuleProperty> properties = new ArrayList<ModuleProperty>();
			ModuleDescription description = new ModuleDescription("Duplicate", "Fork", properties);
			supportedModules.put(description.getType(), description);
		}
		{
			List<ModuleProperty> properties = new ArrayList<ModuleProperty>();
			ModuleDescription description = new ModuleDescription("Accumulate", "Join", properties);
			supportedModules.put(description.getType(), description);
		}
		{
			List<ModuleProperty> properties = new ArrayList<ModuleProperty>();
			ModuleDescription description = new ModuleDescription("Split", "Fork", properties);
			supportedModules.put(description.getType(), description);
		}
		{
			List<ModuleProperty> properties = new ArrayList<ModuleProperty>();
			ModuleDescription description = new ModuleDescription("Concat", "Join", properties);
			supportedModules.put(description.getType(), description);
		}
		{
			List<ModuleProperty> properties = new ArrayList<ModuleProperty>();
			properties.add(new ModuleProperty("Input planes", "noInputPlanes"));
			properties.add(new ModuleProperty("Output planes", "noOutputPlanes"));
			properties.add(new ModuleProperty("Kernel width", "kernelWidth"));
			properties.add(new ModuleProperty("Kernel height", "kernelHeight"));
			ModuleDescription description = new ModuleDescription("Convolution", "Convolution", properties);
			supportedModules.put(description.getType(), description);
		}
		{
			List<ModuleProperty> properties = new ArrayList<ModuleProperty>();
			properties.add(new ModuleProperty("Width", "width"));
			properties.add(new ModuleProperty("Height", "height"));
			ModuleDescription description = new ModuleDescription("MaxPooling", "Convolution", properties);
			supportedModules.put(description.getType(), description);
		}
		{
			List<ModuleProperty> properties = new ArrayList<ModuleProperty>();
			ModuleDescription description = new ModuleDescription("Normalization", "Preprocessing", properties);
			supportedModules.put(description.getType(), description);
		}
	}
	
	@Override
	public Module createModule(TensorFactory factory, Dictionary<String, ?> config)
			throws InstantiationException {
		AbstractModule module = null;
		
		// TODO use reflection for this?
		// for now just hard code an if/else for each known module
		String type = (String)config.get("module.type");
		UUID id = UUID.fromString((String)config.get("module.id"));

		switch(type){
		case "Linear":
			int input = Integer.parseInt((String)config.get("module.linear.input"));
			int output = Integer.parseInt((String)config.get("module.linear.output"));
			
			module = new Linear(factory, id, input, output);
			break;
		case "Tanh":
			module = new Tanh(factory, id);
			break;
		case "Sigmoid":
			module = new Sigmoid(factory, id);
			break;
		case "Softmax":
			module = new Softmax(factory, id);
			break;
		case "ReLU":
			module = new ReLU(factory, id);
			break;
		case "Threshold":
			float thresh = Float.parseFloat((String)config.get("module.threshold.thresh"));
			float val = Float.parseFloat((String)config.get("module.threshold.val"));
			
			module = new Threshold(factory, id, thresh, val);
			break;
		case "Input":
			module = new InputImpl(factory, id);
			break;
		case "Output":
			module = new OutputImpl(factory, id);
			break;
		case "Duplicate":
			module = new Duplicate(factory, id);
			break;
		case "Accumulate":
			module = new Accumulate(factory, id);
			break;
		case "Split":
			module = new Split(factory, id);
			break;
		case "Concat":
			module = new Concat(factory, id);
			break;
		case "Convolution":
			int noInputPlanes = Integer.parseInt((String)config.get("module.convolution.noInputPlanes"));
			int noOutputPlanes = Integer.parseInt((String)config.get("module.convolution.noOutputPlanes"));
			int kernelWidth = Integer.parseInt((String)config.get("module.convolution.kernelWidth"));
			int kernelHeight = Integer.parseInt((String)config.get("module.convolution.kernelHeight"));

			module = new SpatialConvolution(factory, id, noInputPlanes, noOutputPlanes, kernelWidth, kernelHeight);
			break;
		case "MaxPooling":
			int width = Integer.parseInt((String)config.get("module.maxpooling.width"));
			int height = Integer.parseInt((String)config.get("module.maxpooling.height"));

			module = new SpatialMaxPooling(factory, id, width, height);
			break;
		case "Normalization":
			module = new Normalization(factory, id);
			break;
		default:
			throw new InstantiationException("Could not instantiate module of type "+type);
		}
		
		// re-use a cached threadpool
		module.setExecutorService(executor);
		
		return module;
	}

	@Override
	public List<ModuleDescription> getAvailableModules() {
		return new ArrayList<ModuleDescription>(supportedModules.values());
	}

	@Override
	public ModuleDescription getModuleDescription(String name) {
		return supportedModules.get(name);
	}
	
}
