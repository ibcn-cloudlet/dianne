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
 * The DianneOutputs interface provides an API to send output from neural network
 * Output modules to real things (devices) that can actuate upon this.
 * 
 * @author tverbele
 *
 */
public interface DianneOutputs {
	
	/**
	 * List the available things that can act as output
	 * 
	 * @return Descriptions of the available outputs
	 */
	List<OutputDescription> getAvailableOutputs();
	
	/**
	 * Connect the output of an Output module to an output thing.
	 * 
	 * @param nnId ID of the neural network instance
	 * @param outputId module ID of the neural network Output module
	 * @param output the real output to connect this Output module to
	 */
	void setOutput(UUID nnId, UUID outputId, String output);
	
	/**
	 * Disconnect the output of an Output module from an output thing.
	 * 
	 * @param nnId ID of the neural network instance
	 * @param outputId module ID of the neural network Output module
	 * @param output the real output to disconnect this Output module from
	 */
	void unsetOutput(UUID nnId, UUID outputId, String output);
}
