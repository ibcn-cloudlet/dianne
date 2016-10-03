package be.iminds.iot.dianne.rl.agent.strategy.config;

public class GaussianNoiseConfig {

	/**
	 * Max noise
	 */
	public double noiseMax = 1e0;
	
	/**
	 * Min noise
	 */
	public double noiseMin = 0;
	
	/**
	 * Noise exponential decay rate
	 */
	public double noiseDecay = 1e-6;
	
	/**
	 * Minimum value
	 */
	public float min = -1;
	
	/**
	 * Maximum value
	 */
	public float max = 1;
}
