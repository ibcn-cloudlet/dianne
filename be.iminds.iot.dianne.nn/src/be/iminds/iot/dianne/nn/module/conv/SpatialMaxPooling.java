package be.iminds.iot.dianne.nn.module.conv;

import java.util.UUID;

import be.iminds.iot.dianne.nn.module.AbstractModule;
import be.iminds.iot.dianne.tensor.TensorFactory;

public class SpatialMaxPooling extends AbstractModule {
	
	private int w;
	private int h;
	
	public SpatialMaxPooling(TensorFactory factory, 
			int width, int height){
		super(factory);
		this.w = width;
		this.h = height;
	}
	
	public SpatialMaxPooling(TensorFactory factory, UUID id,
			 int width, int height){
		super(factory, id);
		this.w = width;
		this.h = height;
	}

	@Override
	protected void forward() {
		int noPlanes = input.size(0);
		int y = input.size(1)/h;
		int x = input.size(2)/w;
		if(output==null || !output.hasDim(noPlanes, y, x)){
			output = factory.createTensor(noPlanes, y, x);
		}
		
		for(int i=0;i<noPlanes;i++){
			factory.getTensorMath().maxpool2D(output.select(0, i), input.select(0,i), w, h);
		}
	}

	@Override
	protected void backward() {
		if(gradInput == null || !gradInput.sameDim(input)){
			gradInput = factory.createTensor(input.dims());
		}
		int noPlanes = input.size(0);
		for(int i=0;i<noPlanes;i++){
			factory.getTensorMath().dmaxpool2D(gradInput.select(0, i), 
					gradOutput.sameDim(input)? gradOutput.select(0, i):gradOutput, input.select(0, i), w, h);
		}
	}
	
}
