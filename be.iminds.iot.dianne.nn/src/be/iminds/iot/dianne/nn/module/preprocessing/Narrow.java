package be.iminds.iot.dianne.nn.module.preprocessing;

import java.util.UUID;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.nn.module.AbstractModule;
import be.iminds.iot.dianne.api.nn.module.Preprocessor;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;

public class Narrow extends AbstractModule {

	// narrow ranges - should match the dimensions though
	private final int[] ranges;
	
	public Narrow(TensorFactory factory, final int... ranges){
		super(factory);
		this.ranges = ranges;
	}
	
	public Narrow(TensorFactory factory, UUID id, final int... ranges){
		super(factory, id);
		this.ranges = ranges;
	}

	@Override
	protected void forward() {
		output = input.narrow(ranges);
	}

	@Override
	protected void backward() {
		// not implemented
	}

}
