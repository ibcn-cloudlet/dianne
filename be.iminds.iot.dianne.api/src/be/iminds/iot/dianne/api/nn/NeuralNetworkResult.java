package be.iminds.iot.dianne.api.nn;

import be.iminds.iot.dianne.tensor.Tensor;

/**
 * Collects the output of a neural network module: the Tensor and an optional array of tags
 * 
 * @author tverbele
 *
 */
public class NeuralNetworkResult {

	public Tensor tensor;
	public String[] tags;
	
	public NeuralNetworkResult(Tensor tensor, String... tags){
		this.tensor = tensor;
		this.tags = tags;
	}
}
