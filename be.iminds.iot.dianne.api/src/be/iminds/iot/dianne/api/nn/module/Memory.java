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
 * The Memory Module is a special module for modeling memory. This module
 * is used for storing state for recurrent neural networks. It will not automatically
 * forward its output to the next module, but should be triggered externally. This 
 * way, inputs can be skipped for having more long term memory, and it can be used
 * to build and train recurrent neural networks.  
 * 
 * @author tverbele
 *
 */
public interface Memory extends Module {

	/**
	 * Trigger the memory to forward output to the next Module
	 */
	public void triggerForward(final String... tags);
	
	/**
	 * Trigger the memory to backward gradInput to the previous Module
	 */
	public void triggerBackward(final String... tags);
	
	/**
	 * Reset the memory to initial state (and zero out any gradOut)
	 */
	public void reset(int batchSize);

	default void reset(){
		reset(0);
	}
	
	/**
	 * Get the current raw memory data
	 * @return current memory Tensor
	 */
	public Tensor getMemory();

	/**
	 * Push new data into the memory of this module
	 * @param memory
	 */
	public void setMemory(final Tensor memory);
	
}
