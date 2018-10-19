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
package be.iminds.iot.dianne.api.namespace;

public final class DianneNamespace {
	
	/**
	 * Namespace name for neural network capabilities
	 */
	public static final String	DIANNE_NAMESPACE = "dianne.nn";

	/**
	 * String with the input type the neural network expects
	 * 
	 * e.g. "image"
	 */
	public static final String	INPUT = "input";

	/**
	 * Potential properties to indicate supported input dimensions (Long type)
	 */
	public static final String INPUT_WIDTH = "input.width";
	public static final String INPUT_HEIGHT = "input.height";
	public static final String INPUT_DEPTH = "input.depth";
	public static final String INPUT_SIZE = "input.size";

	/**
	 * String indicating necesarry input preprocessing
	 */
	public static final String INPUT_PREPROCESSING = "input.preprocessing";
	
	/**
	 * String with the output type the neural network provides
	 * 
	 * e.g. "classification","regression",...
	 */
	public static final String	OUTPUT = "ouput";

	/**
	 * Potential properties to indicate supported output dimensions (Long type)
	 */
	public static final String OUTPUT_WIDTH = "output.width";
	public static final String OUTPUT_HEIGHT = "output.height";
	public static final String OUTPUT_DEPTH = "output.depth";
	public static final String OUTPUT_SIZE = "output.size";
	
	/**
	 * The dataset it was trained on
	 */
	public static final String	DATASET = "dataset";
	
	/**
	 * String indicating the task of the neural network
	 */
	public static final String	TASK = "task";
	
	private DianneNamespace() {
		// empty
	}
}

