package be.iminds.iot.dianne.api.nn.module;

import be.iminds.iot.dianne.tensor.Tensor;

public interface Input extends Module {

	void input(final Tensor input, final String... tags);
	
}
