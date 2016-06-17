package be.iminds.iot.dianne.nn.util.random;

public abstract class DistributionFactory {

	public abstract Distribution nextDistribution(long seed);
	
	public abstract Distribution nextDistribution();
}
