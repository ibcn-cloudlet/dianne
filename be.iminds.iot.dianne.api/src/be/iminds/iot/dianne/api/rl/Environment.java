package be.iminds.iot.dianne.api.rl;

import be.iminds.iot.dianne.tensor.Tensor;

/**
 * The Environment implements the system an RL Agent is trying to control. It
 * maintains an internal state which is (partially) visible trough observations.
 * Agents performing actions on the Environment will change this state.
 * 
 * @author smbohez
 *
 */
public interface Environment {

	/**
	 * Perform an action on the Environment and receive the associated return
	 * given the current state.
	 * 
	 * @param action the action to be performed on the environment
	 * @return the reward received for performing that action
	 */
	float performAction(final Tensor action);

	/**
	 * Get an observation of the current state of the environment.
	 * 
	 * @return an observation of the current state
	 */
	Tensor getObservation();

}
