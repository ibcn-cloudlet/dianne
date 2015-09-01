package be.iminds.iot.dianne.rl.agent;

import be.iminds.iot.dianne.tensor.Tensor;

public interface ActionStrategy {

	Tensor selectActionFromOutput(Tensor output, long i);
	
}
