package be.iminds.iot.dianne.nn.module;

import be.iminds.iot.dianne.tensor.Tensor;

public interface InputListener {

	public void onBackward(Tensor gradInput);
	
}
