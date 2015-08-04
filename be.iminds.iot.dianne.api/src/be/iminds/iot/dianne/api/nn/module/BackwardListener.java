package be.iminds.iot.dianne.api.nn.module;

import be.iminds.iot.dianne.tensor.Tensor;

public interface BackwardListener {

	void onBackward(final Tensor gradInput, final String... tags);
}
