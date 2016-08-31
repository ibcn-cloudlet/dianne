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
 *     Elias De Coninck
 *******************************************************************************/
package be.iminds.iot.dianne.nn.learn.sampling;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.stream.DoubleStream;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.nn.learn.SamplingStrategy;
import be.iminds.iot.dianne.nn.learn.sampling.config.WeightedUniformConfig;

public class WeightedUniformSamplingStrategy implements SamplingStrategy{

	private final Random random = new Random(System.currentTimeMillis());
	private final Dataset dataset;
	private final int[] startIndices;
	private final double[] probabilities;
	
	public WeightedUniformSamplingStrategy(Dataset dataset, WeightedUniformConfig config) {
		this.dataset = dataset;
		
		double pOther = 1.0 - DoubleStream.of(config.probabilities).sum();
		int noSamplesOther = this.dataset.size();
		for (int i=0; i<config.weightedRanges.length; i+=2)
			noSamplesOther -= (config.weightedRanges[i+1] - config.weightedRanges[i] + 1); //bounds included
		
		List<Integer> startIndices = new ArrayList<>();
		List<Double> probabilities = new ArrayList<>();
		double cumulativeProbability = 0.0;
		if (config.weightedRanges.length > 0 && config.weightedRanges[0] != 0) {
			startIndices.add(0);
			int noSamplesForRange = config.weightedRanges[0];
			cumulativeProbability += pOther*noSamplesForRange/noSamplesOther;
			probabilities.add(cumulativeProbability);
		}
		for (int i=0; i+1<config.weightedRanges.length; i+=2) {
			// start known range
			startIndices.add(config.weightedRanges[i]);
			cumulativeProbability += config.probabilities[i/2];
			probabilities.add(cumulativeProbability);
		
			// start other range
			startIndices.add(config.weightedRanges[i+1] + 1);
			int nextIndice = (i+2 < config.weightedRanges.length) ? config.weightedRanges[i+2] : this.dataset.size();
			int noSamplesForRange = nextIndice  - config.weightedRanges[i+1] - 1;
			cumulativeProbability += pOther*noSamplesForRange/noSamplesOther;
			probabilities.add(cumulativeProbability);
		}
				
		this.startIndices = startIndices.stream().mapToInt(i -> i).toArray();
		this.probabilities = probabilities.stream().mapToDouble(d -> d).toArray();
	}
	
	@Override
	public int next() {
		double p = this.random.nextDouble();
		double prevProb = 0.0;
		double newProb = 0.0;
		for (int i=0; i< this.probabilities.length; i++) {
			prevProb = newProb;
			newProb = this.probabilities[i];
			if (p < newProb) {
				return selectIndexFromRange(i, prevProb, newProb, p);
			}
		}
		return -1; // should never happen!!!
	}

	private int selectIndexFromRange(int i, double pStart, double pEnd, double p) {
		int rangeStart = this.startIndices[i];
		int rangeEnd = (i+1 < this.startIndices.length) ? this.startIndices[i+1] : this.dataset.size();
		return ((int) Math.floor((p-pStart) * (rangeEnd - rangeStart)/(pEnd-pStart)))+rangeStart; // scaled p from range [pStart,pEnd[ to [rangeStart, rangeEnd[
	}

}
