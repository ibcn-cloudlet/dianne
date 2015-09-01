package be.iminds.iot.dianne.rl.agent.api;

import be.iminds.iot.dianne.tensor.Tensor;

public interface ManualActionController {

	public void setAction(Tensor a);
	
}
