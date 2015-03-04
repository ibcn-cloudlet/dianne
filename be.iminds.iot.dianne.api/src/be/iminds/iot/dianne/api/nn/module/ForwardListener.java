package be.iminds.iot.dianne.api.nn.module;

import be.iminds.iot.dianne.tensor.Tensor;

public interface ForwardListener {

	public void onForward(Tensor output);
	
}
