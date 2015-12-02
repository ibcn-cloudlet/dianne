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
 * The Output Module is a special Module that marks the end of a neural network chain.
 * 
 * This is the end point of a neural network, which often is a resulting 1D Tensor classifying
 * the input data. The Output can also give access to a number of labels corresponding with the
 * possible output classes.
 * 
 * @author tverbele
 *
 */
public interface Output extends Module {

	/**
	 * Get the last output
	 * @return the output data
	 */
	Tensor getOutput();

	/**
	 * Get the last output tags
	 * @return the tags
	 */
	String[] getTags();
	
	/**
	 * Get the labels of the output this neural network was trained for
	 * @return output labels
	 */
	String[] getOutputLabels();
	
	/**
	 * Set the labels of the output
	 * @param labels new output labels
	 */
	void setOutputLabels(String[] labels);
	
	/**
	 * Initiate a backward pass with a gradient of the output.
	 * 
	 * @param gradOutput the gradient on the output
	 * @param tags optional tags to tag this backward pass
	 */
	void backpropagate(Tensor gradOutput, String... tags);
	
}
