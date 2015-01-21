package be.iminds.iot.dianne.nn.module;

import be.iminds.iot.dianne.tensor.Tensor;

public interface OutputListener {

	public void onForward(Tensor output);
	
}
