package be.iminds.iot.dianne.rl.environment.kuka.config;

public class FetchCanContinuousConfig {

	/**
	 * Threshold velocity below which the grip action is triggered
	 */
	public float stopThreshold = 0.01f;
	
	/**
	 * Whether magnitude and sign of velocity are seperate inputs
	 */
	public boolean seperateMagnitude = false;
}
