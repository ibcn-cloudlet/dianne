package be.iminds.iot.dianne.nn.util.random.distributions;

import be.iminds.iot.dianne.nn.util.random.Distribution;

public class ExponentialDistribution extends Distribution{

	private final double lambda;
	
	public ExponentialDistribution(long seed, double lambda){
		super(seed);
		this.lambda = lambda;
	}
	
	public ExponentialDistribution(double lambda){
		super();
		this.lambda = lambda;
	}

	@Override
	public double nextDouble() {
		return -Math.log(random.nextDouble())/this.lambda;
	}

}
