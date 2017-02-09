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
package be.iminds.iot.dianne.tensor;

import java.util.Arrays;

/**
 * Represents an n-dimensional tensor in Java
 * 
 * The actual implementation should be done in native code
 * 
 * 
 * @author tverbele
 *
 */
public class Tensor {

	public long address;
	
	public Tensor(){
		this(null, null);
	}
	
	public Tensor(int... dims) {
		this(null, dims);
	}

	public Tensor(int d0, int[] dims){
		int[] d = new int[dims.length+1];
		d[0] = d0;
		System.arraycopy(dims, 0, d, 1, dims.length);
		this.address = init(null, d);
	}
	
	public Tensor(float[] data, int... dims) {
		this.address = init(data, dims);
	}
	
	private Tensor(long address){
		this.address = address;
	}
	
	/**
	 * @return the number of dimensions of this tensor
	 */
	public native int dim();

	/**
	 * @return the dimensions of this tensor
	 */
	public native int[] dims();
	
	/**
	 * @return the total size of the tensor
	 */
	public native int size();

	/**
	 * the size of the d'th dimension
	 * @param d the dimension to query the size
	 * @return the size of the dimension
	 */
	public native int size(final int d);
	
	/**
	 * reshape the dimensions of this tensor, the underlying data remains the same
	 */
	public native void reshape(final int... d);
	
	public void reshape(final int[] d, final int df){
		int[] dn = Arrays.copyOf(d, d.length+1);
		dn[d.length] = df;
		reshape(dn);
	}
	
	/** 
	 * get a value of the tensor
	 * @param d indices of the element
	 * @return the element specified by the index
	 */
	public native float get(final int... d);
	
	/**
	 * get (a copy of) the raw data for this tensor, this way that the tensor 
	 * can be reconstructed with the createTensor(data, dims) factory method
	 */
	public native float[] get();
	
	/**
	 * set a value of the tensor
	 * @param v the new value
	 * @param d the indices of the element to set
	 */
	public native void set(final float v, final int... d);
	
	/**
	 *  copy a complete array of raw data into this tensor
	 */
	public native void set(final float[] data);
	
	/**
	 * fill with fixed value
	 * @param v the new value
	 */
	public native void fill(final float v);

	/**
	 * fill with random values uniformely distributed between 0 and 1
	 */
	public native void rand();

	/**
	 * fill with random values Gaussian ("normally") distributed with mean 0.0 and standard deviation 1.0
	 */
	public native void randn();
	
	/**
	 * fill with 0 or 1 sampled using Bernoulli distribution with 0 <= p <= 1
	 */
	public native void bernoulli(float p);
	
	/**
	 * check if other tensor has same dimensions
	 */
	public native boolean sameDim(final Tensor other);
	
	/**
	 * check if other tensor has these dimensions
	 */
	public native boolean hasDim(final int... dims);
	
	/**
	 * clone this tensor into other tensor, create new one if null or different number of elements
	 * @param other the tensor to clone into
	 * @return the cloned tensor
	 */
	public native Tensor copyInto(final Tensor other);
	
	/**
	 * clone this tensor - creates a deep copy of this tensor
	 */
	public Tensor clone(){
		return copyInto(null);
	}
	
	/**
	 * Return a subtensor narrowing dimension dim from index to index+size-1
	 */
	public native Tensor narrow(final int dim, final int index, final int size);
	
	/**
	 * Return a subtensor narrowing according to the ranges array. This is interpreted
	 * as narrowing dimension 1 from ranges[0] with size ranges[1], narrowing dimension 2 from
	 * ranges[2] with size ranges[3], etc.
	 */
	public Tensor narrow(final int... ranges){
		Tensor n = this;
		for(int i=0;i<ranges.length-1;i+=2){
			n = n.narrow(i/2, ranges[i], ranges[i+1]); 
		}
		return n;
	}
	
	/**
	 * Return a slice at the given index in dimension dim, dimension dim will be removed
	 */
	public native Tensor select(final int dim, final int index);
	
	/**
	 * calculate the transpose of the tensor
	 */
	public native Tensor transpose(Tensor res, final int d1, final int d2);
	
	/**
	 * return the diag vec of the tensor
	 */
	public native Tensor diag(Tensor res);
	
	public boolean equals(Object other){
		if(other == null)
			return false;
		else if(!(other instanceof Tensor))
			return false;
		else
			return equals((Tensor) other);
	}
	
	/**
	 * return whether two tensors are equal (note: they have to be the same type to be equal!)
	 * @param other object to compare to
	 * @return true if the other object represents an equal tensor
	 */
	public boolean equals(Tensor other){
		if(other == null)
			return false;
		else if(other.address == this.address)
			return true;
		else
			return equals(other, 0.0f);
	}
	
	/**
	 * equals with threshold (note: they have to be the same type to be equal!)
	 * @param other object to compare to
	 * @return true if the other object represents an equal tensor with values within threshold range
	 */
	public boolean equals(Tensor other, float threshold){
		if(!this.sameDim(other))
			return false;
		else
			return equalsData(other, threshold);
	}
	
	@Override
	public int hashCode(){
		return (int)address;
	}
	
	@Override
	public String toString(){
		StringBuilder b = new StringBuilder();
		b.append(Arrays.toString(dims()));

		float[] data = get();
		if(data.length > 20){
			b.append(Arrays.toString(Arrays.copyOf(data, 20)));
			b.insert(b.length()-1, "...");
		} else {
			b.append(Arrays.toString(data));
		}
		b.append(" Min: ").append(TensorOps.min(this))
		.append(" Mean: ").append(TensorOps.mean(this))
		.append(" Max: ").append(TensorOps.max(this));
		
		return b.toString();
	}
	
	public void finalize(){
		free();
	}
	
	private native long init(float[] data, int[] dims);
	
	private native void free();
	
	private native boolean equalsData(Tensor other, float threshold);
	
}
