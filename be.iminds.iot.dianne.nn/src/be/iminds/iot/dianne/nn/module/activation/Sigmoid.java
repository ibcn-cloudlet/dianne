package be.iminds.iot.dianne.nn.module.activation;

import be.iminds.iot.dianne.nn.module.AbstractModule;

public class Sigmoid extends AbstractModule {

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
