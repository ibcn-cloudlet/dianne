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
		
		weights = factory.createTensor(outSize, inSize);
		bias = factory.createTensor(outSize);
		
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
		// TODO accumulate instead of replace!
		gradWeights = factory.getTensorMath().vv(gradWeights, gradOutput, input);
		gradBias = gradOutput.clone(gradBias);
	}

}
