package be.iminds.iot.dianne.nn.util.random.distributions;

import be.iminds.iot.dianne.nn.util.random.Distribution;

public class GeometricDistribution extends Distribution{
	
	private final double succes;

	public GeometricDistribution(long seed, double succes){
		super(seed);
		this.succes = succes;
	}
	
	public GeometricDistribution(double succes){
		super();
		this.succes = succes;
	}
	
	@Override
	public double nextDouble() {
		return Math.floor(Math.log(1.0 - random.nextDouble())/Math.log(1.0 - succes) + 1.0);
	}

}
