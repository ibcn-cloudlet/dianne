package be.iminds.iot.dianne.nn.util.random.distributions;

import be.iminds.iot.dianne.nn.util.random.Distribution;

public class UniformDistribution extends Distribution{

	private final double min;
	private final double max;
	
	public UniformDistribution(long seed, double min, double max){
		super(seed);
		this.min = min;
		this.max = max;
	}

	
	public UniformDistribution(double min, double max){
		super();
		this.min = min;
		this.max = max;
	}

	@Override
	public double nextDouble() {
		return this.min+(this.max-this.min)*random.nextDouble();
	}

}
