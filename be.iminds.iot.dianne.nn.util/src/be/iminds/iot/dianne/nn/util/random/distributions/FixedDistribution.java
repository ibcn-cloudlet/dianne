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
 *     Steven Bohez
 *******************************************************************************/
package be.iminds.iot.dianne.nn.util.random.distributions;

import be.iminds.iot.dianne.nn.util.random.Distribution;

public class FixedDistribution extends Distribution{
	
	private final double[] probs;
	private final double[] values;
	
	public FixedDistribution(long seed, double[] probs, double[] values){
		super(seed);
		this.probs = probs;
		this.values = values;
	}
	
	public FixedDistribution(double[] probs, double[] values){
		super();
		this.probs = probs;
		this.values = values;
	}
	
	@Override
	public double nextDouble() {
		double rand = random.nextDouble();
		double total = 0.0;
		int i = 0;
		while((total+=probs[i]) < rand){
			i++;
		}
		return values[i];
	}
}
