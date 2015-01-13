package be.iminds.iot.dianne.nn.module.layer;

import be.iminds.iot.dianne.nn.module.AbstractTrainableModule;
import be.iminds.iot.dianne.tensor.Tensor;

public class Linear extends AbstractTrainableModule {

	private Tensor weights;
	private Tensor bias;
	
	private Tensor gradWeights;
	private Tensor gradBias;
	
	public Linear(int inSize, int outSize){
		super();
		
		parameters = factory.createTensor(outSize, inSize+1);
		gradParameters = factory.createTensor(outSize, inSize+1);
		
		weights = parameters.narrow(1, 0, inSize);
		bias = parameters.narrow(1, inSize, 1);

		gradWeights = gradParameters.narrow(1, 0, inSize);
		gradBias = gradParameters.narrow(1, inSize, 1);
				
		weights.rand();
		bias.rand();
	}
	
	@Override
	protected void forward() {
		output = factory.getTensorMath().addmv(output, bias, weights, input);
	}

	@Override
	protected void backward() {
		gradInput = factory.getTensorMath().tmv(gradInput, weights, gradOutput);
	}

	@Override
	public void accGradParameters() {
		gradWeights = factory.getTensorMath().addvv(gradWeights, gradWeights, gradOutput, input);
		gradBias = factory.getTensorMath().add(gradBias, gradBias, gradOutput);
	}

}
