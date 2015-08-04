package be.iminds.iot.dianne.api.nn.module;

import be.iminds.iot.dianne.tensor.Tensor;

public interface Output extends Module {

	Tensor getOutput();

	String[] getTags();
	
	String[] getOutputLabels();
	
	void setOutputLabels(String[] labels);
	
	void backpropagate(Tensor gradOutput, String... tags);
	
}
