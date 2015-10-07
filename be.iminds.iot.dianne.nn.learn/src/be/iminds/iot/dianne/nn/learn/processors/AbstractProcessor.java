package be.iminds.iot.dianne.nn.learn.processors;

import be.iminds.iot.dianne.api.log.DataLogger;
import be.iminds.iot.dianne.api.nn.NeuralNetwork;
import be.iminds.iot.dianne.api.nn.learn.Processor;
import be.iminds.iot.dianne.tensor.TensorFactory;

public abstract class AbstractProcessor implements Processor {

	protected final TensorFactory factory;
	
	protected final NeuralNetwork nn;
	
	protected final DataLogger logger;
	
	public AbstractProcessor(TensorFactory factory, 
			NeuralNetwork nn,
			DataLogger logger){
		this.factory = factory;
		this.nn = nn;
		this.logger = logger;
	}

	@Override
	public abstract float processNext();
}
