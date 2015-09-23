package be.iminds.iot.dianne.nn.learn.processors;

import java.util.Map;
import java.util.UUID;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.log.DataLogger;
import be.iminds.iot.dianne.api.nn.learn.Processor;
import be.iminds.iot.dianne.api.nn.module.Input;
import be.iminds.iot.dianne.api.nn.module.Output;
import be.iminds.iot.dianne.api.nn.module.Trainable;
import be.iminds.iot.dianne.api.nn.platform.NeuralNetwork;
import be.iminds.iot.dianne.tensor.TensorFactory;

public abstract class AbstractProcessor implements Processor {

	protected final TensorFactory factory;
	
	protected final NeuralNetwork nn;
	
	protected final Dataset dataset;
	
	protected final Map<String, String> config;
	
	protected final DataLogger logger;
	
	public AbstractProcessor(TensorFactory factory, 
			NeuralNetwork nn,
			Dataset dataset, 
			Map<String, String> config,
			DataLogger logger){
		this.factory = factory;
		this.nn = nn;
		this.dataset = dataset;
		
		this.config = config;
		
		this.logger = logger;
	}

	@Override
	public abstract float processNext();
}
