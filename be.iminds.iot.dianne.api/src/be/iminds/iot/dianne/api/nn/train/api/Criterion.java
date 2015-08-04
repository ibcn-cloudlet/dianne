package be.iminds.iot.dianne.api.nn.train.api;

import be.iminds.iot.dianne.tensor.Tensor;

public interface Criterion {

	Tensor forward(final Tensor output, final Tensor target);
	
	Tensor backward(final Tensor output, final Tensor target);
	
}
