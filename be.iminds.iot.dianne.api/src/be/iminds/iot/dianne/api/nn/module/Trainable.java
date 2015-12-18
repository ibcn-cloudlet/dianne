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
package be.iminds.iot.dianne.api.nn.module;

import be.iminds.iot.dianne.tensor.Tensor;

/**
 * The Trainable interface is implemented by Modules that can be trained.
 * 
 * It provides access to the module parameters (often denoted as weights in neural networks),
 * as well as the last calculated gradient on the parameters.
 * 
 * @author tverbele
 *
 */
public interface Trainable extends Module {

	/**
	 * Accumulate the gradient on the parameters
	 */
	void accGradParameters();
	
	/**
	 * Reset the delta on the parameters to zero
	 */
	void zeroDeltaParameters();

	/**
	 * Add the current delta to the parameters
	 */
	void updateParameters();
	
	/**
	 * Add the current delta to the parameters, scaled by factor scale
	 * 
	 * Useful for simple gradient descent training procedures 
	 * (use negative scale for gradient descent, positive scale for gradient ascent)
	 *  
	 * @param scale a scale factor for delta parameters
	 */
	void updateParameters(final float scale);
	
	/**
	 * Return the current delta on the parameters. This can be accumulated gradients 
	 * by calls to accGradParameters, and/or some custom operations on delta parameters.
	 * 
	 * Attention: at the moment this returns a reference to the gradient of the parameters,
	 * only use and change if you know what you are doing.
	 * 
	 * @return the delta on the parameters
	 */
	Tensor getDeltaParameters();
	
	/**
	 * Set new delta parameters for this Module. These deltas are copied.
	 * @param deltaParameters new deltas on the parameters
	 */
	void setDeltaParameters(final Tensor deltaParameters);
	
	/**
	 * Return the current parameters
	 * 
	 * Attention: at the moment this returns a reference to the parameters,
	 * only use and change if you know what you are doing.
	 * 
	 * @return the parameters
	 */
	Tensor getParameters();
	
	/**
	 * Set new parameters for this Module. These parameters are copied.
	 * @param parameters new parameters
	 */
	void setParameters(final Tensor parameters);
	
	/**
	 * Set parameters with random values
	 */
	void randomize();

}
