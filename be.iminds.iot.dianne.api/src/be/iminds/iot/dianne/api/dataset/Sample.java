package be.iminds.iot.dianne.api.dataset;

import be.iminds.iot.dianne.tensor.Tensor;

/**
 * A helper class for representing one sample of a dataset, a combination of
 * an input and output Tensor
 * 
 * @author tverbele
 *
 */
public class Sample {
	
	public final Tensor input;
	public final Tensor output;
	
	public Sample(Tensor i, Tensor o){
		this.input = i;
		this.output = o;
	}
}
