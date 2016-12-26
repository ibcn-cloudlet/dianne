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
package be.iminds.iot.dianne.api.dataset;

import be.iminds.iot.dianne.tensor.Tensor;

/**
 * A helper class for representing one sample of a dataset, a combination of
 * an input and target Tensor
 * 
 * @author tverbele
 *
 */
public class Sample {
	
	public Tensor input;
	public Tensor target;
	
	public Sample(){}
	
	public Sample(Tensor input, Tensor target){
		this.input = input;
		this.target = target;
	}
	
	public Tensor getInput(){
		return input;
	}
	
	public Tensor getTarget(){
		return target;
	}
	
	@Override
	public String toString(){
		StringBuilder b = new StringBuilder();
		b.append("Input: ")
		.append(input)
		.append(" - Target: ")
		.append(target);
		return b.toString();
	}
	
	public Sample copyInto(Sample other){
		if(other == null){
			other = new Sample();
		}
		other.input = input.copyInto(other.input);
		other.target = target.copyInto(other.target);
		return other;
	}
	
	public Sample clone(){
		return copyInto(null);
	}
}
