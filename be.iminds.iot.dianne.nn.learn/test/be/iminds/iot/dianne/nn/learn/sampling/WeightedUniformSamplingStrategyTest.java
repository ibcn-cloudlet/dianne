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

import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

import org.junit.Before;
import org.junit.Test;
import org.mockito.Mockito;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.nn.learn.sampling.config.WeightedUniformConfig;
import junit.framework.Assert;

public class WeightedUniformSamplingStrategyTest {
	
	private Dataset dataset;
	private WeightedUniformConfig config;
	private static final int DATASET_SIZE=2000;
	
	@Before
	public void init() {
		this.config = new WeightedUniformConfig();
		this.dataset = Mockito.mock(Dataset.class);
		Mockito.when(dataset.size()).thenReturn(DATASET_SIZE);
	}
	
	private int[] testChanceRange(int begin, int end, double chance) {
		this.config.weightedRanges = new int[]{begin,end};
		this.config.probabilities = new double[]{chance};
		WeightedUniformSamplingStrategy strat = new WeightedUniformSamplingStrategy(dataset, config);
		return strat.next(5000);
	}
	
	@Test
	public void testZeroChanceRange() {
		int begin = 100;
		int end = DATASET_SIZE - 100;
		for (int i : testChanceRange(begin, end, 0.0)) {
			Assert.assertEquals("Error, value (" + i + ") is between range with 0.0 chance.", true, (0<=i && i<begin) || (end<i && i<DATASET_SIZE));
		}
	}
	
	@Test
	public void testOneChanceRange() {
		int begin = 500;
		int end = DATASET_SIZE - 1000;
		for (int i : testChanceRange(begin, end, 1.0)) {
			Assert.assertEquals("Error, value (" + i + ") is between range with 0.0 chance.", true, begin<=i && i<=end);
		}
	}
	
	@Test
	public void testMultipleRanges() {
		int SAMPLES = 10000000;
		this.config.weightedRanges = new int[]{25,25,326,1000,1002,1003,1050,1500};
		this.config.probabilities = new double[]{0.06,0.30,0.30,0.25};
		WeightedUniformSamplingStrategy strat = new WeightedUniformSamplingStrategy(dataset, config);
		int[] noNumbersRange = new int[this.config.weightedRanges.length/2+1];
		for (int i=0; i<this.config.weightedRanges.length; i+=2) {
			noNumbersRange[i/2]=this.config.weightedRanges[i+1]-this.config.weightedRanges[i]+1;
		}
		noNumbersRange[noNumbersRange.length-1] = DATASET_SIZE - IntStream.of(noNumbersRange).sum();
		int[] indices = strat.next(SAMPLES);
		int[] count = new int[this.config.weightedRanges.length/2+1];
		int[] totalDistr = new int[DATASET_SIZE];
		for (int i : indices) {
			if (25<=i && i<=25) {
				count[0]++;
			} else if (326<=i && i<=1000) {
				count[1]++;
			} else if (1002<=i && i<=1003) {
				count[2]++;
			} else if (1050<=i && i<=1500) {
				count[3]++;
			} else {
				count[4]++;
			}
			totalDistr[i]++;
		}
		for (int i=0; i<count.length; i++) {
			double p = ((double)count[i])/SAMPLES;
			System.out.format("Range %d with p=%f and a total of %d samples\n",i,p,count[i]);
			if (i>=this.config.probabilities.length)
				Assert.assertEquals(1 - DoubleStream.of(this.config.probabilities).sum(), p, 0.01);
			else
				Assert.assertEquals(this.config.probabilities[i], p, 0.01);
		}
		for (int i=0; i<totalDistr.length; i++) {
			System.out.printf("\t%d", i);
		}
		System.out.print("\ncount");
		for (int i=0; i<totalDistr.length; i++) {
			System.out.printf("\t%d", totalDistr[i]);
		}
		System.out.print("\nchance");
		double[] chance = new double[totalDistr.length];
		for (int i=0; i<totalDistr.length; i++) {
			double p = ((double)totalDistr[i])/SAMPLES;
			System.out.printf("\t%.5f", p);
			chance[i]=p;
		}
		System.out.print("\n?=");
		double[] expected = new double[totalDistr.length];
		for (int i=0; i<totalDistr.length; i++) {
			if (25<=i && i<=25) {
				expected[i]=this.config.probabilities[0]/noNumbersRange[0];
			} else if (326<=i && i<=1000) {
				expected[i]=this.config.probabilities[1]/noNumbersRange[1];
			} else if (1002<=i && i<=1003) {
				expected[i]=this.config.probabilities[2]/noNumbersRange[2];
			} else if (1050<=i && i<=1500) {
				expected[i]=this.config.probabilities[3]/noNumbersRange[3];
			} else {
				expected[i]=(1 - DoubleStream.of(this.config.probabilities).sum())/noNumbersRange[4];
			}
			System.out.printf("\t%.5f", expected[i]);
		}
		System.out.print("\ncount?=");
		for (int i=0; i<totalDistr.length; i++) {
			double pExpected;
			if (25<=i && i<=25) {
				pExpected=this.config.probabilities[0]/noNumbersRange[0];
			} else if (326<=i && i<=1000) {
				pExpected=this.config.probabilities[1]/noNumbersRange[1];
			} else if (1002<=i && i<=1003) {
				pExpected=this.config.probabilities[2]/noNumbersRange[2];
			} else if (1050<=i && i<=1500) {
				pExpected=this.config.probabilities[3]/noNumbersRange[3];
			} else {
				pExpected=(1 - DoubleStream.of(this.config.probabilities).sum())/noNumbersRange[4];
			}
			System.out.printf("\t%.0f", pExpected*SAMPLES);
		}
		System.out.println();
		for (int i=0;i<chance.length; i++) {
			Assert.assertEquals(String.format("Chance for indice %d has more than 0.00001 difference.", i), expected[i], chance[i], 0.001);
		}
	}
}
