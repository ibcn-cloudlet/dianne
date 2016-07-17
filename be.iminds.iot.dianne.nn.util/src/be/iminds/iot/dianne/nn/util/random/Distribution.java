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
package be.iminds.iot.dianne.nn.util.random;

import java.util.Random;

public abstract class Distribution {
	
	protected final Random random;
	
	public Distribution(long seed){
		random = new HighQualityRandom(seed);
	}
	
	public Distribution(){
		random = new HighQualityRandom();
	}
	
	public abstract double nextDouble();
	
	public long nextLong(){
		return (long) this.nextDouble();
	}
	
	public int nextInt(){
		return (int) this.nextDouble();
	}
}
