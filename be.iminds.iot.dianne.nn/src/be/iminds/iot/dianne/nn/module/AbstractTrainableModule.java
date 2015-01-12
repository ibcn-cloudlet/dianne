package be.iminds.iot.dianne.nn.module;

import be.iminds.iot.dianne.tensor.Tensor;

public abstract class AbstractTrainableModule extends AbstractModule implements Trainable {

	protected Tensor parameters;
	protected Tensor gradParameters;
	
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
