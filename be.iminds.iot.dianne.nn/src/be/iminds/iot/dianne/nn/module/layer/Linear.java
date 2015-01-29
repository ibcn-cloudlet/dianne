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
		parameters = factory.createTensor(outSize, inSize+1);
		gradParameters = factory.createTensor(outSize, inSize+1);
		
		weights = parameters.narrow(1, 0, inSize);
		bias = parameters.narrow(1, inSize, 1);

		gradWeights = gradParameters.narrow(1, 0, inSize);
		gradBias = gradParameters.narrow(1, inSize, 1);
				
		weights.randn();
		bias.randn();
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
