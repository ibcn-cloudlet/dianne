package be.iminds.iot.dianne.nn.module;

import java.util.UUID;

import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;

public abstract class AbstractTrainableModule extends AbstractModule implements Trainable {

	protected Tensor parameters;
	protected Tensor gradParameters;
	
	public AbstractTrainableModule(TensorFactory factory) {
		super(factory);
	}
	
	public AbstractTrainableModule(TensorFactory factory, UUID id) {
		super(factory, id);
	}
	
	@Override
	public abstract void accGradParameters();

	@Override
	public void zeroGradParameters() {
		gradParameters.fill(0.0f);
	}

	@Override
	public void updateParameters(float learningRate) {
		factory.getTensorMath().add(parameters, parameters, learningRate, gradParameters);
	}

}
