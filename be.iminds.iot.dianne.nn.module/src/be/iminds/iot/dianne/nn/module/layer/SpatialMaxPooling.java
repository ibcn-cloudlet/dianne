package be.iminds.iot.dianne.nn.module.layer;

import java.util.UUID;

import be.iminds.iot.dianne.api.nn.module.AbstractModule;
import be.iminds.iot.dianne.tensor.TensorFactory;

public class SpatialMaxPooling extends AbstractModule {
	
	private int w;
	private int h;
	private int sx;
	private int sy;
	
	public SpatialMaxPooling(TensorFactory factory, 
			int width, int height, int sx, int sy){
		super(factory);
		this.w = width;
		this.h = height;
		this.sx = sx;
		this.sy = sy;
	}
	
	public SpatialMaxPooling(TensorFactory factory, UUID id,
			 int width, int height, int sx, int sy){
		super(factory, id);
		this.w = width;
		this.h = height;
		this.sx = sx;
		this.sy = sy;
	}

	@Override
	protected void forward() {
		int noPlanes = input.size(0);
		int y = input.size(1)/h;
		int x = input.size(2)/w;
		if(output==null || !output.hasDim(noPlanes, y, x)){
			output = factory.createTensor(noPlanes, y, x);
		}
		
		output = factory.getTensorMath().spatialmaxpool(output, input, w, h, sx, sy);
	}

	@Override
	protected void backward() {	
		if(gradInput == null || !gradInput.sameDim(input)){
			gradInput = factory.createTensor(input.dims());
		}

		factory.getTensorMath().spatialdmaxpool(gradInput, gradOutput, input, w, h, sx, sy);
	}
	
}
