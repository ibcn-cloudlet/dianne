package be.iminds.iot.dianne.nn.util.random.distributions;

import be.iminds.iot.dianne.nn.util.random.Distribution;
import be.iminds.iot.dianne.nn.util.random.DistributionFactory;

public class ExponentialDistributionFactory extends DistributionFactory{

	private final Distribution lambda;
	
	public ExponentialDistributionFactory(Distribution lambda){
		super();
		this.lambda = lambda;
	}
	
	@Override
	public Distribution nextDistribution(long seed) {
		return new ExponentialDistribution(seed,lambda.nextDouble());
	}
	
	@Override
	public Distribution nextDistribution() {
		return new ExponentialDistribution(lambda.nextDouble());
	}
}
