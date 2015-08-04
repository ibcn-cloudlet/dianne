package be.iminds.iot.dianne.api.nn.module;

import be.iminds.iot.dianne.tensor.Tensor;

public interface ForwardListener {

	void onForward(final Tensor output, final String... tags);
	
}
