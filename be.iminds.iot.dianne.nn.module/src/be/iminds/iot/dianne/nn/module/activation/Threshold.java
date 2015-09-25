package be.iminds.iot.dianne.nn.module.activation;

import java.util.UUID;

import be.iminds.iot.dianne.api.nn.module.AbstractModule;
import be.iminds.iot.dianne.tensor.TensorFactory;

public class Threshold extends AbstractModule {
	
	private final float thresh;
	private final float val;
	
	public Threshold(TensorFactory factory, float thresh, float val) {
		super(factory);
		this.thresh = thresh;
		this.val = val;
	}
	
	public Threshold(TensorFactory factory, UUID id, float thresh, float val) {
		super(factory, id);
		this.thresh = thresh;
		this.val = val;
	}

	@Override
	protected void forward() {
		output = factory.getTensorMath().thresh(output, input, thresh, 0, val);
	}

	@Override
	protected void backward() {
		gradInput = factory.getTensorMath().cmul(gradInput, gradOutput, 
				factory.getTensorMath().dthresh(gradInput, input, thresh, 0));
	}

}
