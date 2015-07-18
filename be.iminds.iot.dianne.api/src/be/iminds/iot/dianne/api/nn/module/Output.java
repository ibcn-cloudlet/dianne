package be.iminds.iot.dianne.api.nn.module;

import be.iminds.iot.dianne.tensor.Tensor;

public interface Output extends Module {

	public Tensor getOutput();

	public String[] getTags();
	
	public String[] getOutputLabels();
	
	public void setOutputLabels(String[] labels);
	
	public void backpropagate(Tensor gradOutput, String... tags);
	
}
