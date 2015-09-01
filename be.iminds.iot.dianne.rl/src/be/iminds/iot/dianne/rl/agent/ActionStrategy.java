package be.iminds.iot.dianne.rl.agent;

import java.util.Map;

import be.iminds.iot.dianne.tensor.Tensor;

public interface ActionStrategy {

	Tensor selectActionFromOutput(Tensor output, long i);
	
	void configure(Map<String, String> config);
}
