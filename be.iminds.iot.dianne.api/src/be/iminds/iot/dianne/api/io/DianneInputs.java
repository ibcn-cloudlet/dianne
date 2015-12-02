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
package be.iminds.iot.dianne.api.io;

import java.util.List;
import java.util.UUID;

/**
 * The DianneInputs interface provides an API to couple real inputs from things 
 * (devices) to actual neural network Input modules.
 * 
 * @author tverbele
 *
 */
public interface DianneInputs {

	/**
	 * List the available things that can act as input
	 * 
	 * @return Descriptions of the available inputs
	 */
	List<InputDescription> getAvailableInputs();
	
	/**
	 * Connect a real input to an Input module
	 * 
	 * @param nnId ID of the neural network instance
	 * @param inputId module ID of the neural network Input module
	 * @param input the real input to connect to this Input
	 */
	void setInput(UUID nnId, UUID inputId, String input);
	
	/**
	 * Disconnect a real input from an Input module
	 * 
	 * @param nnId ID of the neural network instance
	 * @param inputId module ID of the neural network Input module
	 * @param input the real input to disconnect this Input from
	 */
	void unsetInput(UUID nnId, UUID inputId, String input);
	
}
