package be.iminds.iot.dianne.api.rl;

import be.iminds.iot.dianne.tensor.Tensor;

/**
 * Notify a listener when an Action is performed on the Environment
 * 
 * Useful for creating a UI for the Environment
 * 
 * Use the target property on the EnvironmentListener service to specify which specific
 * Environment to listen to
 * 
 * @author tverbele
 *
 */
public interface EnvironmentListener {

	/**
	 * Called when an action is executed
	 * 
	 * This method is called synchronously, thus a UI can slow down the Agent
	 * for better visualization
	 * 
	 * @param reward the reward resulting from this action
	 * @param nextState the next state tensor
	 */
	public void onAction(float reward, Tensor nextState);
	
}
