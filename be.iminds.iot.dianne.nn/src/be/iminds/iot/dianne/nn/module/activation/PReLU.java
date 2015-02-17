package be.iminds.iot.dianne.nn.module.activation;

import java.util.UUID;

import be.iminds.iot.dianne.nn.module.AbstractTrainableModule;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;

public class PReLU extends AbstractTrainableModule{
	
	public PReLU(TensorFactory factory) {
		this(factory, 0.25f);
	}
	
	public PReLU(TensorFactory factory, float init) {
		super(factory);
		init(init);
	}
	
	public PReLU(TensorFactory factory, UUID id) {
		this(factory, id, 0.25f);
	}

	public PReLU(TensorFactory factory, UUID id, float init) {
		super(factory, id);
		init(init);
	}
	
	private void init(float init) {
		parameters = factory.createTensor(1);
		gradParameters = factory.createTensor(1);
		
		parameters.set(init, 0);
	}
	
	@Override
	protected void forward() {
		output = factory.getTensorMath().thresh(output, input, 0f, parameters.get(0), 0f);
	}

	@Override
	protected void backward() {
		gradInput = factory.getTensorMath().cmul(gradInput, gradOutput,
				factory.getTensorMath().dthresh(gradInput, input, 0f, parameters.get(0)));
	}

	private Tensor temp;
	
	@Override
	public void accGradParameters() {
		temp = factory.getTensorMath().mul(temp, input, -1f);
		temp = factory.getTensorMath().thresh(temp, temp, 0f, 0f, 0f);
		temp = factory.getTensorMath().mul(temp, temp, -1f);
		
		temp = factory.getTensorMath().cmul(temp, temp, gradOutput);
		gradParameters = factory.getTensorMath().add(gradParameters, gradParameters,
				factory.getTensorMath().sum(temp));
	}
}
