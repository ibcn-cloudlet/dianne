package be.iminds.iot.dianne.nn.module.activation;

import java.util.UUID;

import be.iminds.iot.dianne.tensor.TensorFactory;

public class ReLU extends Threshold {

	public ReLU(TensorFactory factory) {
		super(factory, 0f, 0f);
	}

	public ReLU(TensorFactory factory, UUID id) {
		super(factory, id, 0f, 0f);
	}
	
}
