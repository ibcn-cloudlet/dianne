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

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import be.iminds.iot.dianne.tensor.Tensor;

/**
 * A helper class for representing a sequence of samples/batches of a dataset
 * 
 * @author tverbele
 *
 */
public class Sequence<T extends Sample> implements Iterable <T>{

	public int size;
	public List<T> data;
	
	public Sequence(){
		size = 0;
		data = new ArrayList<T>();
	}

	public Sequence(List<T> data){
		this.size = data.size();
		this.data = new ArrayList<T>(data);
	}
	
	public Sequence(List<T> data, int size){
		if(data.size() < size){
			throw new InstantiationError("Not enough items in the data");
		}
		this.size = size;
		this.data = new ArrayList<T>(data);
	}
	
	public int size(){
		return size;
	}
	
	public T get(int index){
		if(index >= size){
			throw new ArrayIndexOutOfBoundsException();
		}
		return data.get(index);
	}
	
	public Tensor getInput(int index){
		return get(index).input;
	}
	
	public Tensor getTarget(int index){
		return get(index).target;
	}
	
	@Override
	public String toString(){
		StringBuilder b = new StringBuilder();
		for(int i=0;i<size;i++){
			b.append("[").append(i).append("] ")
			.append(data.get(i));
		}
		return b.toString();
	}
	
	public Sequence<T> copyInto(Sequence<T> other){
		if(other==null){
			other = new Sequence<T>();
		}
		for(int i=0;i<size;i++){
			if(other.data.size() > i){
				// copy into element
				data.get(i).copyInto(other.data.get(i));
			} else {
				// add new element
				other.data.add((T)data.get(i).clone());
			}
		}
		other.size = size;
		return other;
	}
	
	public Sequence<T> clone(){
		return copyInto(null);
	}

	@Override
	public Iterator<T> iterator() {
		return new Iterator<T>() {
			private int i = 0;
			
			@Override
			public boolean hasNext() {
				return i < size-1;
			}

			@Override
			public T next() {
				return data.get(i++);
			}
		};
	}
}
