package be.iminds.iot.dianne.nn.util.random.distributions;

import be.iminds.iot.dianne.nn.util.random.Distribution;

public class NormalDistribution extends Distribution {

	private final double mu;
	private final double sigma;
	
	public NormalDistribution(long seed, double mu, double sigma) {
		super(seed);
		this.mu = mu;
		this.sigma = sigma;
	}
	
	public NormalDistribution(double mu, double sigma) {
		super();
		this.mu = mu;
		this.sigma = sigma;
	}

	@Override
	public double nextDouble() {
		return random.nextGaussian()*sigma + mu;
	}

}
