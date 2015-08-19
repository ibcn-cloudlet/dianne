package be.iminds.iot.dianne.nn.learn.processors;

import java.util.Map;
import java.util.UUID;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.nn.learn.Processor;
import be.iminds.iot.dianne.api.nn.module.Input;
import be.iminds.iot.dianne.api.nn.module.Output;
import be.iminds.iot.dianne.api.nn.module.Trainable;
import be.iminds.iot.dianne.tensor.TensorFactory;

public abstract class AbstractProcessor implements Processor {

	final TensorFactory factory;
	
	final Input input;
	final Output output;
	final Map<UUID, Trainable> toTrain;
	
	final Dataset dataset;
	
	final Map<String, String> config;
	
	public AbstractProcessor(TensorFactory factory, 
			Input input, 
			Output output, 
			Map<UUID, Trainable> toTrain, 
			Dataset dataset, 
			Map<String, String> config){
		this.factory = factory;
		this.input = input;
		this.output = output;
		this.toTrain = toTrain;
		this.dataset = dataset;
		
		this.config = config;
	}

	@Override
	public abstract float processNext();
}
