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
import java.util.Random;

import be.iminds.iot.dianne.api.dataset.SequenceDataset;
import be.iminds.iot.dianne.nn.util.DianneConfigHandler;
import be.iminds.iot.dianne.rnn.learn.sampling.config.UniformSequenceSamplingStrategyConfig;

public class UniformSequenceSamplingStrategy implements SequenceSamplingStrategy{

	private Random random = new Random(System.currentTimeMillis());
	
	private UniformSequenceSamplingStrategyConfig config;
	
	private final SequenceDataset<?,?> dataset;
	
	public UniformSequenceSamplingStrategy(SequenceDataset<?,?> dataset, Map<String, String> config) {
		this.dataset = dataset;
		this.config = DianneConfigHandler.getConfig(config, UniformSequenceSamplingStrategyConfig.class);
	}

	@Override
	public int sequence() {
		return random.nextInt(dataset.sequences());
	}

	@Override
	public int next(int sequence, int length) {
		if(config.fromStart)
			return 0;
		
		int sequenceLength = dataset.sequenceLength(sequence);
		int range = sequenceLength - length;
		if(range < 0)
			throw new RuntimeException("Cannot sample sequence of length "+length+" from dataset sequence");
		
		return random.nextInt(range);
	}
	
	
}
