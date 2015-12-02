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

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.tensor.Tensor;

/**
 * A Preprocessor is a special Module that performs some preprocessing on the input data.
 * 
 * In order to work correctly, this Module often requires to have access to the complete dataset.
 * For example, a Normalization module will estimate parameters from example inputs in a dataset.
 * 
 * @author tverbele
 *
 */
public interface Preprocessor extends Module {

	/**
	 * Generate the parameters using the given dataset
	 * @param data dataset to use for calculating the parameters
	 */
	void preprocess(Dataset data);

	/**
	 * Get the parameters from this Module
	 * @return current parameters
	 */
	Tensor getParameters();
	
	/**
	 * Manually set the parameters for this Module
	 * @param parameters new parameters
	 */
	void setParameters(final Tensor parameters);
	
	/**
	 * Returns if the data is already preprocessed or if the parameters are set
	 */
	boolean isPreprocessed(); 
}
