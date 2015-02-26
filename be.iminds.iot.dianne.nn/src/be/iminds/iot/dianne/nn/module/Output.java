package be.iminds.iot.dianne.nn.module;

import be.iminds.iot.dianne.tensor.Tensor;

public interface Output extends Module {

	public Tensor getOutput();

	public String[] getOutputLabels();
	
	public void setOutputLabels(String[] labels);
	
	public void backpropagate(Tensor gradOutput);
	
}
