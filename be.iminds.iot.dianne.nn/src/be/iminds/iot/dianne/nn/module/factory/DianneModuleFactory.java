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

import be.iminds.iot.dianne.api.nn.module.AbstractModule;
import be.iminds.iot.dianne.api.nn.module.Module;
import be.iminds.iot.dianne.api.nn.module.description.ModuleProperty;
import be.iminds.iot.dianne.api.nn.module.description.ModuleType;
import be.iminds.iot.dianne.api.nn.module.factory.ModuleFactory;
import be.iminds.iot.dianne.nn.module.activation.ReLU;
import be.iminds.iot.dianne.nn.module.activation.Sigmoid;
import be.iminds.iot.dianne.nn.module.activation.Softmax;
import be.iminds.iot.dianne.nn.module.activation.Tanh;
import be.iminds.iot.dianne.nn.module.activation.Threshold;
import be.iminds.iot.dianne.nn.module.fork.Duplicate;
import be.iminds.iot.dianne.nn.module.fork.Split;
import be.iminds.iot.dianne.nn.module.io.InputImpl;
import be.iminds.iot.dianne.nn.module.io.OutputImpl;
import be.iminds.iot.dianne.nn.module.join.Accumulate;
import be.iminds.iot.dianne.nn.module.join.Concat;
import be.iminds.iot.dianne.nn.module.layer.Linear;
import be.iminds.iot.dianne.nn.module.layer.SpatialConvolution;
import be.iminds.iot.dianne.nn.module.layer.SpatialMaxPooling;
import be.iminds.iot.dianne.nn.module.preprocessing.Normalization;
import be.iminds.iot.dianne.tensor.TensorFactory;

@Component(property={"aiolos.export=false"})
public class DianneModuleFactory implements ModuleFactory {

	private ExecutorService executor = Executors.newCachedThreadPool();
	
	private final Map<String, ModuleType> supportedModules = new HashMap<String, ModuleType>();
	
	@Activate
	public void activate(){
		// build list of supported modules
		// TODO use reflection for this?
		
		{
			List<ModuleProperty> properties = new ArrayList<ModuleProperty>();
			properties.add(new ModuleProperty("Input size", "input"));
			properties.add(new ModuleProperty("Output size", "output"));
			ModuleType type = new ModuleType("Linear", "Layer", properties, true);
			supportedModules.put(type.getType(), type);
		}
		{
			List<ModuleProperty> properties = new ArrayList<ModuleProperty>();
			ModuleType type = new ModuleType("Sigmoid", "Activation", properties, false);
			supportedModules.put(type.getType(), type);
		}
		{
			List<ModuleProperty> properties = new ArrayList<ModuleProperty>();
			ModuleType type = new ModuleType("Tanh", "Activation", properties, false);
			supportedModules.put(type.getType(), type);
		}
		{
			List<ModuleProperty> properties = new ArrayList<ModuleProperty>();
			ModuleType type = new ModuleType("Softmax", "Activation", properties, false);
			supportedModules.put(type.getType(), type);
		}		
		{
			List<ModuleProperty> properties = new ArrayList<ModuleProperty>();
			ModuleType type = new ModuleType("ReLU", "Activation", properties, false);
			supportedModules.put(type.getType(), type);
		}
		{
			List<ModuleProperty> properties = new ArrayList<ModuleProperty>();
			properties.add(new ModuleProperty("Threshold value", "thresh"));
			properties.add(new ModuleProperty("Replacement value", "val"));
			ModuleType type = new ModuleType("Threshold", "Activation", properties, false);
			supportedModules.put(type.getType(), type);
		}
		{
			List<ModuleProperty> properties = new ArrayList<ModuleProperty>();
			ModuleType type = new ModuleType("Input", "Input-Output", properties, false);
			supportedModules.put(type.getType(), type);
		}
		{
			List<ModuleProperty> properties = new ArrayList<ModuleProperty>();
			ModuleType type = new ModuleType("Output", "Input-Output", properties, false);
			supportedModules.put(type.getType(), type);
		}
		{
			List<ModuleProperty> properties = new ArrayList<ModuleProperty>();
			ModuleType type = new ModuleType("Duplicate", "Fork", properties, false);
			supportedModules.put(type.getType(), type);
		}
		{
			List<ModuleProperty> properties = new ArrayList<ModuleProperty>();
			ModuleType type = new ModuleType("Accumulate", "Join", properties, false);
			supportedModules.put(type.getType(), type);
		}
		{
			List<ModuleProperty> properties = new ArrayList<ModuleProperty>();
			ModuleType type = new ModuleType("Split", "Fork", properties, false);
			supportedModules.put(type.getType(), type);
		}
		{
			List<ModuleProperty> properties = new ArrayList<ModuleProperty>();
			ModuleType type = new ModuleType("Concat", "Join", properties, false);
			supportedModules.put(type.getType(), type);
		}
		{
			List<ModuleProperty> properties = new ArrayList<ModuleProperty>();
			properties.add(new ModuleProperty("Input planes", "noInputPlanes"));
			properties.add(new ModuleProperty("Output planes", "noOutputPlanes"));
			properties.add(new ModuleProperty("Kernel width", "kernelWidth"));
			properties.add(new ModuleProperty("Kernel height", "kernelHeight"));
			properties.add(new ModuleProperty("Stride X", "strideX"));
			properties.add(new ModuleProperty("Stride Y", "strideY"));	
			properties.add(new ModuleProperty("Full (add zero padding)", "full"));	
			ModuleType type = new ModuleType("Convolution", "Layer", properties, true);
			supportedModules.put(type.getType(), type);
		}
		{
			List<ModuleProperty> properties = new ArrayList<ModuleProperty>();
			properties.add(new ModuleProperty("Width", "width"));
			properties.add(new ModuleProperty("Height", "height"));
			properties.add(new ModuleProperty("Stride X", "strideX"));
			properties.add(new ModuleProperty("Stride Y", "strideY"));			
			ModuleType type = new ModuleType("MaxPooling", "Layer", properties, true);
			supportedModules.put(type.getType(), type);
		}
		{
			List<ModuleProperty> properties = new ArrayList<ModuleProperty>();
			ModuleType type = new ModuleType("Normalization", "Preprocessing", properties, false);
			supportedModules.put(type.getType(), type);
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
			
			int strideX = hasProperty(config,"module.convolution.strideX") ? Integer.parseInt((String)config.get("module.convolution.strideX")) : 1;
			int strideY = hasProperty(config,"module.convolution.strideY") ? Integer.parseInt((String)config.get("module.convolution.strideY")) : 1;

			boolean full = hasProperty(config,"module.convolution.full") ? Boolean.parseBoolean((String)config.get("module.convolution.full")) : false;

			module = new SpatialConvolution(factory, id, noInputPlanes, noOutputPlanes, kernelWidth, kernelHeight, strideX, strideY, full);
			break;
		case "MaxPooling":
			int width = Integer.parseInt((String)config.get("module.maxpooling.width"));
			int height = Integer.parseInt((String)config.get("module.maxpooling.height"));

			int sx = hasProperty(config, "module.maxpooling.strideX") ? Integer.parseInt((String)config.get("module.maxpooling.strideX")) : width;
			int sy = hasProperty(config,"module.maxpooling.strideY") ? Integer.parseInt((String)config.get("module.maxpooling.strideY")) : height;
			
			module = new SpatialMaxPooling(factory, id, width, height, sx, sy);
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
	public List<ModuleType> getAvailableModuleTypes() {
		return new ArrayList<ModuleType>(supportedModules.values());
	}

	@Override
	public ModuleType getModuleType(String name) {
		return supportedModules.get(name);
	}
	
	private boolean hasProperty(Dictionary<String, ?> config, String property){
		String value = (String) config.get(property);
		if(value==null){
			return false;
		} else if(value.isEmpty()){
			return false;
		}
		return true;
	}
}
