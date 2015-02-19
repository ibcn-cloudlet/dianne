package be.iminds.iot.dianne.nn.module.conv;

import java.util.UUID;

import be.iminds.iot.dianne.nn.module.AbstractTrainableModule;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;

public class SpatialConvolution extends AbstractTrainableModule {

	private int noInputPlanes;
	private int noOutputPlanes;
	private int kernelWidth;
	private int kernelHeight;
	// TODO strides support?
	
	// subtensors for weights / bias
	Tensor weights;
	Tensor gradWeights;
	Tensor bias;
	Tensor gradBias;
	
	public SpatialConvolution(TensorFactory factory,
			int noInputPlanes, int noOutputPlanes, 
			int kernelWidth, int kernelHeight){
		super(factory);
		init(noInputPlanes, noOutputPlanes, kernelWidth, kernelHeight);
	}
	
	public SpatialConvolution(TensorFactory factory, UUID id,
			int noInputPlanes, int noOutputPlanes, 
			int kernelWidth, int kernelHeight){
		super(factory, id);
		init(noInputPlanes, noOutputPlanes, kernelWidth, kernelHeight);
	}
	
	protected void init(int noInputPlanes, int noOutputPlanes, 
			int kernelWidth, int kernelHeight){
		this.noInputPlanes = noInputPlanes;
		this.noOutputPlanes = noOutputPlanes;
		this.kernelWidth = kernelWidth;
		this.kernelHeight = kernelHeight;
		
		parameters = factory.createTensor(noOutputPlanes*noInputPlanes*kernelWidth*kernelHeight+noOutputPlanes);
		weights = parameters.narrow(0, 0, noOutputPlanes*noInputPlanes*kernelWidth*kernelHeight);
		weights.reshape(noOutputPlanes, noInputPlanes, kernelWidth, kernelHeight);
		bias = parameters.narrow(0, noOutputPlanes*noInputPlanes*kernelWidth*kernelHeight, noOutputPlanes);
		
		gradParameters = factory.createTensor(noOutputPlanes*noInputPlanes*kernelWidth*kernelHeight+noOutputPlanes);
		gradWeights = gradParameters.narrow(0, 0, noOutputPlanes*noInputPlanes*kernelWidth*kernelHeight);
		gradWeights.reshape(noOutputPlanes, noInputPlanes, kernelWidth, kernelHeight);
		gradBias = gradParameters.narrow(0, noOutputPlanes*noInputPlanes*kernelWidth*kernelHeight, noOutputPlanes);
		
		
		// initialize weights uniform [-std, std] with std = 1/sqrt(kW*kH*noInputPlanes)  [from torch]
		parameters.rand();
		float std = (float) (1f/Math.sqrt(kernelWidth*kernelHeight*noInputPlanes));
		parameters = factory.getTensorMath().mul(parameters, parameters, 2*std);
		parameters = factory.getTensorMath().sub(parameters, parameters, std);
	}
	
	@Override
	protected void forward() {
		int[] outDims = new int[3];
		outDims[0] = noOutputPlanes;
		if(input.dim()==2){
			outDims[1] = input.size(0) - kernelHeight + 1;
			outDims[2] = input.size(1) - kernelWidth + 1;
		} else if(input.dim()==3){
			outDims[1] = input.size(1) - kernelHeight+ 1;
			outDims[2] = input.size(2) - kernelWidth + 1;
		} // else error?
		if(output==null || !output.hasDim(outDims)){
			output = factory.createTensor(outDims);
		}
		// TODO check input planes dim? // check kernel sizes?
	
		// TODO create subtensors once and reuse?
		Tensor temp = null;
		
		for(int i=0;i<noOutputPlanes;i++){
			Tensor planeKernels = weights.select(0, i);
			Tensor outputPlane = output.select(0, i);
			outputPlane.fill(0.0f);
			for(int j=0;j<noInputPlanes;j++){
				Tensor kernel = planeKernels.select(0, j);
				
				// TODO convadd operation to avoid temp?
				temp = factory.getTensorMath().convolution2D(temp,
						noInputPlanes== 1 ? input : input.select(0, j), kernel, 1, 1, false, false);
				factory.getTensorMath().add(outputPlane, outputPlane, temp);
			}
			
			// add bias
			factory.getTensorMath().add(outputPlane, outputPlane, bias.get(i));
		}
	}

	@Override
	protected void backward() {
		// backward based on http://andrew.gibiansky.com/blog/machine-learning/convolutional-neural-networks/
		if(gradInput == null || !gradInput.sameDim(input)){
			gradInput = factory.createTensor(input.dims());
		}
		
		// TODO create subtensors once and reuse?
		Tensor temp = null;
		
		for(int i=0;i<noInputPlanes;i++){
			Tensor planeKernels = weights.select(1, i);
			Tensor gradInputPlane = noInputPlanes== 1 ? gradInput : gradInput.select(0, i);
			gradInputPlane.fill(0.0f);
			for(int j=0;j<noOutputPlanes;j++){
				Tensor kernel = planeKernels.select(0, j);
				
				// update gradInput
				// this should be "full" convolution and flipped kernel?
				temp = factory.getTensorMath().convolution2D(temp,
						gradOutput.select(0, j), kernel, 1, 1, true, true);
				factory.getTensorMath().add(gradInputPlane, gradInputPlane, temp);
			}
		}
	}

	@Override
	public void accGradParameters() {
		// calculate grad weights based on http://andrew.gibiansky.com/blog/machine-learning/convolutional-neural-networks/
		
		Tensor temp = null;
		
		for(int i=0;i<noOutputPlanes;i++){
			Tensor planeGradKernels = gradWeights.select(0, i);
		
			for(int j=0;j<noInputPlanes;j++){
				Tensor gradKernel = planeGradKernels.select(0, j);
				
				//  update gradKernel
				temp = factory.getTensorMath().convolution2D(temp, 
						noInputPlanes== 1 ? input : input.select(0, j), gradOutput.select(0, i), 1, 1, false, false);

				factory.getTensorMath().add(gradKernel, gradKernel, temp);
			}
		}
		
		// grad bias
		for(int i=0;i<noOutputPlanes;i++){
			float sum = factory.getTensorMath().sum(gradOutput.select(0, i));
			gradBias.set(gradBias.get(i)+sum, i);
		}
	}
}
