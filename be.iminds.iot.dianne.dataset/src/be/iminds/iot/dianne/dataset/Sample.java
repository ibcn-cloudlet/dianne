package be.iminds.iot.dianne.dataset;

import be.iminds.iot.dianne.tensor.Tensor;

public class Sample {
	public final Tensor input;
	public final Tensor output;
	
	public Sample(Tensor i, Tensor o){
		this.input = i;
		this.output = o;
	}
}
