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
package be.iminds.iot.dianne.api.nn;

import java.util.List;
import java.util.Map;
import java.util.UUID;

import be.iminds.iot.dianne.tensor.Tensor;

/**
 * Collects the output sequence of a neural network: the Tensor(s) list and an optional array of tags
 * 
 * @author tverbele
 *
 */
public class NeuralNetworkSequenceResult {

	public Map<UUID, List<Tensor>> tensors;
	public List<Tensor> tensor;
	public String[] tags;
	
	public NeuralNetworkSequenceResult(Map<UUID, List<Tensor>> tensors, String... tags){
		this.tensors = tensors;
		if(tensors.size() == 1){
			tensor = tensors.values().iterator().next();
		}
		this.tags = tags;
	}
}
