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
package be.iminds.iot.dianne.api.nn.learn;

import be.iminds.iot.dianne.tensor.Tensor;

/**
 * The optimization Criterion for training a neural network.
 * 
 * The grad call should be preceded by a call to loss with same output-target pair
 * 
 * @author tverbele
 *
 */
public interface Criterion {

	/**
	 * Returns the loss when comparing the actual output with a target output
	 * 
	 * @param output the neural network output
	 * @param target the desired target output
	 * @return the loss for the output
	 */
	float loss(final Tensor output, final Tensor target);
	
	/**
	 * Returns gradient feeding into the Output Module backwards for training
	 * 
	 * @param output the neural network output
	 * @param target the desired target output
	 * @return the gradient for backpropagation
	 */
	Tensor grad(final Tensor output, final Tensor target);
	
}
