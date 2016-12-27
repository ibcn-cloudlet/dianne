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
 *     Tim Verbelen, Steven Bohez, Elias De Coninck
 *******************************************************************************/
package be.iminds.iot.dianne.rnn.learn.sampling;

import java.util.Map;

import be.iminds.iot.dianne.api.dataset.SequenceDataset;
import be.iminds.iot.dianne.nn.learn.sampling.SamplingFactory.SamplingConfig;

public class SequenceSamplingFactory {
	
	public static SequenceSamplingStrategy createSamplingStrategy(SamplingConfig strategy, SequenceDataset<?,?> d, Map<String, String> config){
		SequenceSamplingStrategy sampling = null;

		switch(strategy) {
		default:
			sampling = new UniformSequenceSamplingStrategy(d);
		}
		
		return sampling;
	}
}
