package be.iminds.iot.dianne.nn.util.random.distributions;

import be.iminds.iot.dianne.nn.util.random.Distribution;
import be.iminds.iot.dianne.nn.util.random.DistributionFactory;

public class NormalDistributionFactory extends DistributionFactory{

	private final Distribution mu;
	private final Distribution sigma;
	
	public NormalDistributionFactory(Distribution mu, Distribution sigma){
		super();
		this.mu = mu;
		this.sigma = sigma;
	}
	
	@Override
	public Distribution nextDistribution(long seed) {
		return new NormalDistribution(seed,mu.nextDouble(),sigma.nextDouble());
	}

	@Override
	public Distribution nextDistribution() {
		return new NormalDistribution(mu.nextDouble(),sigma.nextDouble());
	}
}
