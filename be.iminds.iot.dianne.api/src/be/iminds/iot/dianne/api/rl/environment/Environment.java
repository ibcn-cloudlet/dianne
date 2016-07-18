/*******************************************************************************
 * DIANNE  - Framework for distributed artificial neural networks
 * Copyright (C) 2015  iMinds - IBCN - UGent
 *
 * This file is part of DIANNE.
 *
 * DIANNE is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * Contributors:
 *     Tim Verbelen, Steven Bohez
 *******************************************************************************/
package be.iminds.iot.dianne.api.rl.environment;

import java.util.Map;

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
	 * @return dimensions of an observation
	 */
	int[] observationDims();
	
	/**
	 * @return dimensions of an action
	 */
	int[] actionDims();
	
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
	 * @param t a tensor to copy the new observation into - in case null is passed a new Tensor is created
	 * @return an observation of the current state
	 */
	Tensor getObservation(Tensor t);

	
	default Tensor getObservation(){
		return getObservation(null);
	}
	
	/**
	 * Reset the environment with the given configuration (if possible).
	 */
	void reset();

	/**
	 * Set up the environment... should be called first before one can peform actions
	 * @param config
	 */
	void setup(Map<String, String> config);
	
	/**
	 * Cleanup the environment when you no longer require it
	 */
	void cleanup();
	
}
