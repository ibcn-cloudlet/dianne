package be.iminds.iot.dianne.nn.util.random.distributions;

import be.iminds.iot.dianne.nn.util.random.Distribution;
import be.iminds.iot.dianne.nn.util.random.DistributionFactory;

public class PositiveDistributionFactory extends DistributionFactory {

	private final DistributionFactory factory;
	
	public PositiveDistributionFactory(DistributionFactory factory){
		super();
		this.factory = factory;
	}
	
	@Override
	public Distribution nextDistribution(long seed) {
		return new PositiveDistribution(factory.nextDistribution(seed));
	}
	
	@Override
	public Distribution nextDistribution() {
		return new PositiveDistribution(factory.nextDistribution());
	}

}
