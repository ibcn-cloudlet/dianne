package be.iminds.iot.dianne.nn.module.activation;

import java.util.UUID;

import be.iminds.iot.dianne.nn.module.AbstractModule;
import be.iminds.iot.dianne.tensor.TensorFactory;

public class LogSoftmax extends AbstractModule {

	public LogSoftmax(TensorFactory factory) {
		super(factory);
	}
	
	public LogSoftmax(TensorFactory factory, UUID id) {
		super(factory, id);
	}
	
	@Override
	protected void forward() {
		output = factory.getTensorMath().logsoftmax(output, input);
	}

	@Override
	protected void backward() {
		float sum = factory.getTensorMath().sum(gradOutput);
		
		gradInput = factory.getTensorMath().sub(gradInput, gradOutput,
				factory.getTensorMath().mul(null,
						factory.getTensorMath().exp(null, output), sum));
	}
	
}
