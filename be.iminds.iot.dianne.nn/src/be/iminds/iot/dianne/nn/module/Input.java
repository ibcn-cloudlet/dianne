package be.iminds.iot.dianne.nn.module;

import be.iminds.iot.dianne.tensor.Tensor;

public interface Input extends Module {

	public void input(Tensor input);
	
	public void addInputListener(InputListener listener);
	
	public void removeInputListener(InputListener listener);
	
}
