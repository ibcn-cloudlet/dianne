package be.iminds.iot.dianne.nn.module.activation;

import java.util.UUID;

import be.iminds.iot.dianne.api.nn.module.AbstractModule;
import be.iminds.iot.dianne.tensor.TensorFactory;

public class Sigmoid extends AbstractModule {

	public Sigmoid(TensorFactory factory) {
		super(factory);
	}
	
	public Sigmoid(TensorFactory factory, UUID id) {
		super(factory, id);
	}
	
	@Override
	protected void forward() {
		output = factory.getTensorMath().sigmoid(output, input);
	}

	@Override
	protected void backward() {
		gradInput = factory.getTensorMath().cmul(gradInput, gradOutput, 
				factory.getTensorMath().dsigmoid(gradInput, output));
	}

}
