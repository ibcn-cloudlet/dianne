package be.iminds.iot.dianne.nn.module.conv;

import java.util.Arrays;
import java.util.UUID;

import be.iminds.iot.dianne.nn.module.AbstractTrainableModule;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;

public class SpatialConvolution extends AbstractTrainableModule {

	private int noInputPlanes;
	private int noOutputPlanes;
	private int kernelWidth;
	private int kernelHeight;
	
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
	
	// TODO strides support?
	protected void init(int noInputPlanes, int noOutputPlanes, 
			int kernelWidth, int kernelHeight){
		this.noInputPlanes = noInputPlanes;
		this.noOutputPlanes = noOutputPlanes;
		this.kernelWidth = kernelWidth;
		this.kernelHeight = kernelHeight;
		
		parameters = factory.createTensor(noOutputPlanes, noInputPlanes, kernelWidth, kernelHeight); 
	}
	
	@Override
	protected void forward() {
		if(output==null || !output.hasDim(noOutputPlanes, input.dims()[1]-kernelWidth+1, input.dims()[2]-kernelHeight+1)){
			output = factory.createTensor(noOutputPlanes, input.dims()[1]-kernelWidth+1, input.dims()[2]-kernelHeight+1);
		}
		// TODO check input planes dim?
	
		// TODO create subtensors once and reuse?
		Tensor temp = null;
		
		for(int i=0;i<noOutputPlanes;i++){
			Tensor planeKernels = parameters.select(0, i);
			Tensor outputPlane = output.select(0, i);
			outputPlane.fill(0.0f);
			for(int j=0;j<noInputPlanes;j++){
				Tensor kernel = planeKernels.select(0, j);
				
				// TODO convadd operation to avoid temp?
				temp = factory.getTensorMath().convolution2D(temp, input.select(0, j), kernel);
				factory.getTensorMath().add(outputPlane, outputPlane, temp);
			}
		}
	}

	@Override
	protected void backward() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void accGradParameters() {
		// TODO Auto-generated method stub
		
	}

}
