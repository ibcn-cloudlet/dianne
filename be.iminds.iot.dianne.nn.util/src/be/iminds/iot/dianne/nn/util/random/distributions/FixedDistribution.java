package be.iminds.iot.dianne.nn.util.random.distributions;

import be.iminds.iot.dianne.nn.util.random.Distribution;

public class FixedDistribution extends Distribution{
	
	private final double[] probs;
	private final double[] values;
	
	public FixedDistribution(long seed, double[] probs, double[] values){
		super(seed);
		this.probs = probs;
		this.values = values;
	}
	
	public FixedDistribution(double[] probs, double[] values){
		super();
		this.probs = probs;
		this.values = values;
	}
	
	@Override
	public double nextDouble() {
		double rand = random.nextDouble();
		double total = 0.0;
		int i = 0;
		while((total+=probs[i]) < rand){
			i++;
		}
		return values[i];
	}
}
