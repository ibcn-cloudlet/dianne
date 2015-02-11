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
		output = factory.getTensorMath().maxpool2D(output, input, w, h);
	}

	@Override
	protected void backward() {
		// TODO 
	}
	
}
