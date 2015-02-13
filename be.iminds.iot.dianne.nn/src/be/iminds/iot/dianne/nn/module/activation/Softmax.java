package be.iminds.iot.dianne.nn.module.activation;

import java.util.UUID;

import be.iminds.iot.dianne.nn.module.AbstractModule;
import be.iminds.iot.dianne.tensor.TensorFactory;

public class Softmax extends AbstractModule {

	public Softmax(TensorFactory factory) {
		super(factory);
	}
	
	public Softmax(TensorFactory factory, UUID id) {
		super(factory, id);
	}
	
	@Override
	protected void forward() {
		output = factory.getTensorMath().softmax(output, input);
	}

	@Override
	protected void backward() {
		float sum = factory.getTensorMath().sum(factory.getTensorMath().cmul(null, gradOutput, output));
		
		gradInput = factory.getTensorMath().sub(gradInput, gradOutput, sum);
		gradInput = factory.getTensorMath().cmul(gradInput, output, gradInput);
	}
}
