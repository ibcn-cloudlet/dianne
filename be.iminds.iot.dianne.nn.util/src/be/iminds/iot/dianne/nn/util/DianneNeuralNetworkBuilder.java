package be.iminds.iot.dianne.nn.util;

import java.util.HashMap;
import java.util.LinkedList;
import java.util.Map;
import java.util.UUID;

import be.iminds.iot.dianne.api.nn.module.dto.ModuleDTO;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkDTO;

public class DianneNeuralNetworkBuilder {

	private final String name;
	private final LinkedList<ModuleDTO> modules = new LinkedList<>();
	
	public DianneNeuralNetworkBuilder(String name){
		this.name = name;
		
		ModuleDTO input = new ModuleDTO(UUID.randomUUID(), "Input", null, null, null);
		modules.add(input);
	}
	
	public DianneNeuralNetworkBuilder addLinear(int input, int output){
		Map<String, String> properties = new HashMap<>();
		properties.put("input", ""+input);
		properties.put("output", ""+output);
		
		ModuleDTO prev = modules.getLast();
		ModuleDTO linear = new ModuleDTO(UUID.randomUUID(), "Linear", null, new UUID[]{prev.id}, properties);
		prev.next = new UUID[]{linear.id};
		modules.add(linear);
		
		return this;
	}

	public DianneNeuralNetworkBuilder addConvolutional(int inputPlanes, int outputPlanes, 
			int kernelSize){
		return addConvolutional(inputPlanes, outputPlanes, kernelSize, kernelSize, 1, 1, false);
	}
	
	public DianneNeuralNetworkBuilder addConvolutional(int inputPlanes, int outputPlanes, 
			int kernelWidth, int kernelHeight){
		return addConvolutional(inputPlanes, outputPlanes, kernelWidth, kernelHeight, 1, 1, false);
	}
	
	public DianneNeuralNetworkBuilder addConvolutional(int inputPlanes, int outputPlanes, 
			int kernelWidth, int kernelHeight, int strideX, int strideY, boolean pad){
		Map<String, String> properties = new HashMap<>();
		properties.put("noInputPlanes", ""+inputPlanes);
		properties.put("noOutputPlanes", ""+outputPlanes);
		properties.put("kernelWidth", ""+kernelWidth);
		properties.put("kernelHeight", ""+kernelHeight);
		properties.put("strideX", ""+strideX);
		properties.put("strideY", ""+strideY);
		properties.put("pad", ""+pad);
		
		ModuleDTO prev = modules.getLast();
		ModuleDTO conv = new ModuleDTO(UUID.randomUUID(), "Convolution", null, new UUID[]{prev.id}, properties);
		prev.next = new UUID[]{conv.id};
		modules.add(conv);
		
		return this;
	}
	
	public DianneNeuralNetworkBuilder addMaxpool(int kernelWidth, int kernelHeight, int strideX, int strideY){
		Map<String, String> properties = new HashMap<>();
		properties.put("width", ""+kernelWidth);
		properties.put("height", ""+kernelHeight);
		properties.put("strideX", ""+strideX);
		properties.put("strideY", ""+strideY);
		
		ModuleDTO prev = modules.getLast();
		ModuleDTO maxpool = new ModuleDTO(UUID.randomUUID(), "MaxPooling", null, new UUID[]{prev.id}, properties);
		prev.next = new UUID[]{maxpool.id};
		modules.add(maxpool);
		
		return this;
	}
	
	public DianneNeuralNetworkBuilder addReLU(){
		ModuleDTO prev = modules.getLast();
		ModuleDTO relu = new ModuleDTO(UUID.randomUUID(), "ReLU", null, new UUID[]{prev.id}, null);
		prev.next = new UUID[]{relu.id};
		modules.add(relu);
		return this;
	}
	
	public DianneNeuralNetworkBuilder addSigmoid(){
		ModuleDTO prev = modules.getLast();
		ModuleDTO sigmoid = new ModuleDTO(UUID.randomUUID(), "Sigmoid", null, new UUID[]{prev.id}, null);
		prev.next = new UUID[]{sigmoid.id};
		modules.add(sigmoid);
		return this;
	}
	
	public DianneNeuralNetworkBuilder addTanh(){
		ModuleDTO prev = modules.getLast();
		ModuleDTO tanh = new ModuleDTO(UUID.randomUUID(), "Tanh", null, new UUID[]{prev.id}, null);
		prev.next = new UUID[]{tanh.id};
		modules.add(tanh);
		return this;
	}
	
	public DianneNeuralNetworkBuilder addSoftmax(){
		ModuleDTO prev = modules.getLast();
		ModuleDTO softmax = new ModuleDTO(UUID.randomUUID(), "Softmax", null, new UUID[]{prev.id}, null);
		prev.next = new UUID[]{softmax.id};
		modules.add(softmax);
		return this;
	}
	
	public NeuralNetworkDTO create(){
		ModuleDTO prev = modules.getLast();
		ModuleDTO output = new ModuleDTO(UUID.randomUUID(), "Output", null, new UUID[]{prev.id}, null);
		prev.next = new UUID[]{output.id};
		modules.add(output);
		
		return new NeuralNetworkDTO(name, modules);
	}
	
	/**
	 * Create multi-layer perceptron network
	 * @return
	 */
	public enum Activation {
			ReLU,
			Sigmoid,
			Tanh
	}
	
	public static NeuralNetworkDTO createMLP(String name, int input, int output, Activation activation, int...layers){
		DianneNeuralNetworkBuilder builder = new DianneNeuralNetworkBuilder(name);
		
		for(int i=0;i<layers.length;i++){
			int in = i==0 ? input : layers[i-1];
			int out = layers[i];
			builder.addLinear(in, out);
			
			switch(activation){
			case ReLU:
				builder.addReLU();
				break;
			case Sigmoid:
				builder.addSigmoid();
				break;
			case Tanh:
				builder.addTanh();
				break;
			}
		}

		builder.addLinear(layers[layers.length-1], output);
		builder.addSoftmax();
		return builder.create();
	}
}
