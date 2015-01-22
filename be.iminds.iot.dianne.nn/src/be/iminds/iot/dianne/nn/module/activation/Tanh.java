package be.iminds.iot.dianne.nn.module.activation;

import java.util.UUID;

import be.iminds.iot.dianne.nn.module.AbstractModule;
import be.iminds.iot.dianne.tensor.TensorFactory;

public class Tanh extends AbstractModule {

	public Tanh(TensorFactory factory) {
		super(factory);
	}
	
	public Tanh(TensorFactory factory, UUID id) {
		super(factory, id);
	}
	
	@Override
	protected void forward() {
		output = factory.getTensorMath().tanh(output, input);
	}

	@Override
	protected void backward() {
		// derivative of tanh:
		// dtanh/dx = 1-tanh^2 
		//
		// thus:
		// gradInput = gradOutput * ( dtan/dx(input) )
		//           = gradOutput * (1 - tanh^2(input))
		//           = gradOutput * (1 - output^2)
		gradInput = factory.getTensorMath().cmul(gradInput, gradOutput, 
				factory.getTensorMath().dtanh(gradInput, output));
	}

}
