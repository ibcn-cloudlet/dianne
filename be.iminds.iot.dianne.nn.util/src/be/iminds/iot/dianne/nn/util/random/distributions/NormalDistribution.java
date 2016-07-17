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

public class NormalDistribution extends Distribution {

	private final double mu;
	private final double sigma;
	
	public NormalDistribution(long seed, double mu, double sigma) {
		super(seed);
		this.mu = mu;
		this.sigma = sigma;
	}
	
	public NormalDistribution(double mu, double sigma) {
		super();
		this.mu = mu;
		this.sigma = sigma;
	}

	@Override
	public double nextDouble() {
		return random.nextGaussian()*sigma + mu;
	}

}
