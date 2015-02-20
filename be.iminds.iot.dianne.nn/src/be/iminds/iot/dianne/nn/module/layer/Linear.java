package be.iminds.iot.dianne.nn.module.layer;

import java.util.UUID;

import be.iminds.iot.dianne.nn.module.AbstractTrainableModule;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;

public class Linear extends AbstractTrainableModule {

	private Tensor weights;
	private Tensor bias;
	
	private Tensor gradWeights;
	private Tensor gradBias;
	
	public Linear(TensorFactory factory, int inSize, int outSize){
		super(factory);
		init(inSize, outSize);
	}
	
	public Linear(TensorFactory factory, UUID id, int inSize, int outSize){
		super(factory, id);
		init(inSize, outSize);
	}
	
	private void init(int inSize, int outSize){
		parameters = factory.createTensor(outSize*(inSize+1));
		gradParameters = factory.createTensor(outSize*(inSize+1));
		
		weights = parameters.narrow(0, 0, outSize*inSize);
		weights.reshape(outSize, inSize);
		bias = parameters.narrow(0, outSize*inSize, outSize);
		bias.reshape(outSize);

		gradWeights = gradParameters.narrow(0, 0, outSize*inSize);
		gradWeights.reshape(outSize, inSize);
		gradBias = gradParameters.narrow(0, outSize*inSize, outSize);
		gradBias.reshape(outSize);
		
		// initialize weights uniform [-std, std] with std = 1/sqrt(noInputs)  [from torch]
		parameters.rand();
		float std = (float) (1f/Math.sqrt(inSize));
		parameters = factory.getTensorMath().mul(parameters, parameters, 2*std);
		parameters = factory.getTensorMath().sub(parameters, parameters, std);
		
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
