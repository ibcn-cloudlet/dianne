package be.iminds.iot.dianne.rl.environment.kuka.config;

public class KukaConfig {

	/**
	 * Energy penalization, discouraging the policy to go full speed on the motors. Energy is calculated as the norm of the action tensor.
	 */
	public float energyPenalization = 0f;
}
