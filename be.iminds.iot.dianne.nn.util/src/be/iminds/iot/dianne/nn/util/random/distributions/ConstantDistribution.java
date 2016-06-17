package be.iminds.iot.dianne.nn.util.random.distributions;

import be.iminds.iot.dianne.nn.util.random.Distribution;

public class ConstantDistribution extends Distribution{

	private final double constant;

	//Note: no constructor with seed provided,
	//as this distribution is independent of a seed.
	public ConstantDistribution(double constant){
		super();
		this.constant = constant;
	}

	@Override
	public double nextDouble() {
		return constant;
	}
	
}
