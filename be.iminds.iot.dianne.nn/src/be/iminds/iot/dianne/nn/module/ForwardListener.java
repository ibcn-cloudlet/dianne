package be.iminds.iot.dianne.nn.module;

import be.iminds.iot.dianne.tensor.Tensor;

public interface ForwardListener {

	public void onForward(Tensor output);
	
}
