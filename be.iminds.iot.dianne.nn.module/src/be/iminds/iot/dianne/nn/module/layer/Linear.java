package be.iminds.iot.dianne.nn.module.layer;

import java.util.UUID;

import be.iminds.iot.dianne.api.nn.module.AbstractTrainableModule;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;

public class Linear extends AbstractTrainableModule {

	private int inSize;
	private int outSize;
	
	private Tensor weights;
	private Tensor bias;
	
	private Tensor deltaWeights;
	private Tensor deltaBias;
	
	public Linear(TensorFactory factory, int inSize, int outSize){
		super(factory);
		initParameters(inSize, outSize);
	}
	
	public Linear(TensorFactory factory, UUID id, int inSize, int outSize){
		super(factory, id);
		initParameters(inSize, outSize);
	}
	
	private void initParameters(int inSize, int outSize){
		this.inSize = inSize;
		this.outSize = outSize;
		
		parameters = factory.createTensor(outSize*(inSize+1));
		
		weights = parameters.narrow(0, 0, outSize*inSize);
		weights.reshape(outSize, inSize);
		bias = parameters.narrow(0, outSize*inSize, outSize);
		bias.reshape(outSize);

		parameters.fill(0.0f);
	}
	
	private void initDeltaParameters(){
		deltaParameters = factory.createTensor(outSize*(inSize+1));
		
		deltaWeights = deltaParameters.narrow(0, 0, outSize*inSize);
		deltaWeights.reshape(outSize, inSize);
		deltaBias = deltaParameters.narrow(0, outSize*inSize, outSize);
		deltaBias.reshape(outSize);
		
		deltaParameters.fill(0.0f);
	}
	
	@Override 
	public void randomize(){
		// randomize weights uniform [-std, std] with std = 1/sqrt(noInputs)  [from torch]
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
		if(deltaParameters==null){
			initDeltaParameters();
		}
		gradInput = factory.getTensorMath().tmv(gradInput, weights, gradOutput);
	}

	@Override
	public void accGradParameters() {
		deltaWeights = factory.getTensorMath().addvv(deltaWeights, deltaWeights, gradOutput, input);
		deltaBias = factory.getTensorMath().add(deltaBias, deltaBias, gradOutput);
	}

}
