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
package be.iminds.iot.dianne.nn.learn.sampling;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.nn.learn.SamplingStrategy;

public class PermutationSamplingStrategy implements SamplingStrategy{

	private List<Integer> indices = new ArrayList<>();
	private int current = 0;
	
	public PermutationSamplingStrategy(Dataset dataset) {
		for(int i=0;i<dataset.size();i++){
			indices.add(i);
		}
		Collections.shuffle(indices);
	}
	
	@Override
	public int next() {
		if(current == indices.size()){
			current = 0;
			Collections.shuffle(indices);
		}
		return indices.get(current++);
	}

}
