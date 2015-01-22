package be.iminds.iot.dianne.nn.module.activation;

import be.iminds.iot.dianne.nn.module.AbstractModule;
import be.iminds.iot.dianne.tensor.TensorFactory;

public class Sigmoid extends AbstractModule {

	public Sigmoid(TensorFactory factory) {
		super(factory);
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
