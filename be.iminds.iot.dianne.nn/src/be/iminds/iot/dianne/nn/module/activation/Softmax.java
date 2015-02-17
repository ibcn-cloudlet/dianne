package be.iminds.iot.dianne.nn.module.activation;

import java.util.UUID;

import be.iminds.iot.dianne.nn.module.AbstractModule;
import be.iminds.iot.dianne.tensor.TensorFactory;

public class Softmax extends AbstractModule {

	private float alpha = 0.0001f;
	
	public Softmax(TensorFactory factory) {
		super(factory);
	}
	
	public Softmax(TensorFactory factory, UUID id) {
		super(factory, id);
	}
	
	@Override
	protected void forward() {
		output = factory.getTensorMath().softmax(output, input);
		
		// this makes sure that you don't end up with zeros and a one, which 
		// gives -Inf in the NLL ... this does add a (small) error though...
		output = factory.getTensorMath().add(output, output, alpha);
		output = factory.getTensorMath().div(output, output, 1f + alpha*output.size());
	}

	@Override
	protected void backward() {
		float sum = factory.getTensorMath().sum(factory.getTensorMath().cmul(null, gradOutput, output));
		
		gradInput = factory.getTensorMath().sub(gradInput, gradOutput, sum);
		gradInput = factory.getTensorMath().cmul(gradInput, output, gradInput);
	}
}
