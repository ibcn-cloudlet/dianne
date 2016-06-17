package be.iminds.iot.dianne.nn.util.random.distributions;

import be.iminds.iot.dianne.nn.util.random.Distribution;

public class PositiveDistribution extends Distribution {

	private final Distribution dist;
	
	//Note: no constructor with seed provided,
	//as this is a wrapper class of an actual distribution.
	public PositiveDistribution(Distribution dist){
		super();
		this.dist = dist;
	}
	
	@Override
	public double nextDouble() {
		double result;
		while((result = dist.nextDouble()) < 0);
		return result;
	}
}
