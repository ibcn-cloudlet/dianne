package be.iminds.iot.dianne.nn.module.activation;

import be.iminds.iot.dianne.nn.module.AbstractModule;

public class Sigmoid extends AbstractModule {

	public Sigmoid(){
		// TODO should we allocate tensor here?
		output = factory.createTensor(1);
	}
	
	@Override
	protected void forward() {
		if(!input.sameDim(output)){
			output = factory.createTensor(input.dims());
		}
		output = factory.getTensorMath().sigmoid(output, input);
	}

	@Override
	protected void backward() {
		// TODO check also if dim of output is same as gradoutput?
		if(!gradInput.sameDim(gradOutput)){
			gradInput = factory.createTensor(gradOutput.dims());
		}
		gradInput = factory.getTensorMath().cmul(gradInput, gradOutput, 
				factory.getTensorMath().dsigmoid(gradInput, output));
	}

}
