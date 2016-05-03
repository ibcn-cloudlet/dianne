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

	private long address;
	private int[] dims;
	
	public Tensor(int... dims) {
		this(null, dims);
	}
	
	public Tensor(float[] data, int... dims) {
		this.dims = dims;
		this.address = init(data, dims);
	}
	
	private Tensor(long address) {
		this.address = address;
		this.dims = dims(address);
	}
	
	/**
	 * @return the number of dimensions of this tensor
	 */
	public int dim(){
		return dims.length;
	}

	/**
	 * @return the dimensions of this tensor
	 */
	public int[] dims(){
		return Arrays.copyOf(dims, dims.length);
	}
	
	/**
	 * @return the total size of the tensor
	 */
	public int size(){
		int size = 1;
		for(int i=0;i<dims.length;i++){
			size *= dims[i];
		}
		return size;
	}
	
	/**
	 * the size of the d'th dimension
	 * @param d the dimension to query the size
	 * @return the size of the dimension
	 */
	public int size(final int d){
		return dims[d];
	}
	
	/**
	 * reshape the dimensions of this tensor, the underlying data remains the same
	 */
	public void reshape(final int... d){
		this.dims = d;
		reshape(address, d);
	}
	
	/** 
	 * get a value of the tensor
	 * @param d indices of the element
	 * @return the element specified by the index
	 */
	public float get(final int... d){
		return get(address, d);
	}
	
	/**
	 * get (a copy of) the raw data for this tensor, this way that the tensor 
	 * can be reconstructed with the createTensor(data, dims) factory method
	 */
	public float[] get(){
		return get(address);
	}
	
	/**
	 * set a value of the tensor
	 * @param v the new value
	 * @param d the indices of the element to set
	 */
	public void set(final float v, final int... d){
		set(address, v, d);
	}
	
	/**
	 *  copy a complete array of raw data into this tensor
	 */
	public void set(final float[] data){
		set(address, data);
	}
	
	/**
	 * fill with fixed value
	 * @param v the new value
	 */
	public void fill(final float v){
		fill(address, v);
	}

	/**
	 * fill with random values uniformely distributed between 0 and 1
	 */
	public void rand(){
		rand(address);
	}

	/**
	 * fill with random values Gaussian ("normally") distributed with mean 0.0 and standard deviation 1.0
	 */
	public void randn(){
		randn(address);
	}
	
	/**
	 * fill with 0 or 1 sampled using Bernoulli distribution with 0 <= p <= 1
	 */
	public void bernoulli(float p){
		bernoulli(address, p);
	}
	
	/**
	 * check if other tensor has same dimensions
	 */
	public boolean sameDim(final Tensor other){
		if(dims.length!=other.dims.length){
			return false;
		}
		for(int i=0;i<dims.length;i++){
			if(other.dims[i] != dims[i]){
				return false;
			}
		}
		return true;
	}
	
	/**
	 * check if other tensor has these dimensions
	 */
	public boolean hasDim(final int... dims){
		if(this.dims.length!=dims.length){
			return false;
		}
		for(int i=0;i<dims.length;i++){
			if(this.dims[i] != dims[i]){
				return false;
			}
		}
		return true;
	}
	
	/**
	 * clone this tensor into other tensor, create new one if null or different number of elements
	 * @param other the tensor to clone into
	 * @return the cloned tensor
	 */
	public Tensor copyInto(final Tensor other){
		Tensor copy = other;
		if(copy==null){
			copy = new Tensor(dims);
		} else if(copy.address==this.address){
			return this;
		}
		
		copyInto(address, copy.address);
		return copy;
	}
	
	/**
	 * Return a subtensor narrowing dimension dim from index to index+size-1
	 */
	public Tensor narrow(final int dim, final int index, final int size){
		final long n = narrow(address, dim, index, size);
		return new Tensor(n);
	}
	
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
	public Tensor select(final int dim, final int index){
		final long s = select(address, dim, index);
		return new Tensor(s);
	}
	
	/**
	 * calculate the transpose of the tensor
	 */
	public Tensor transpose(Tensor res, final int d1, final int d2){
		final long l = transpose(address, d1, d2);
		// TH does not do transpose to res vector, just copy here
		Tensor t = new Tensor(l);
		if(res!=null){
			t.copyInto(res);
			return res;
		} else {
			return t;
		}
	}
	
	/**
	 * return the diag vec of the tensor
	 */
	public Tensor diag(Tensor res){
		final long l = diag(address, res == null ?  0 : res.address);
		return res==null ? new Tensor(l) : res;
	}
	
	public boolean equals(Object other){
		if(!(other instanceof Tensor)){
			return false;
		} 
		Tensor o = (Tensor) other;
		if(o.address == address){
			return true;
		}
		return equals(o);
	}
	
	/**
	 * return whether two tensors are equal (note: they have to be the same type to be equal!)
	 * @param other object to compare to
	 * @return true if the other object represents an equal tensor
	 */
	public boolean equals(Tensor other){
		return equals(other, 0.0f);
	}
	
	/**
	 * equals with threshold (note: they have to be the same type to be equal!)
	 * @param other object to compare to
	 * @return true if the other object represents an equal tensor with values within threshold range
	 */
	public boolean equals(Tensor other, float threshold){
		if(!this.sameDim(other)){
			return false;
		}
		return equals(address, other.address, threshold);
	}
	
	@Override
	public int hashCode(){
		return (int)address;
	}
	
	@Override
	public String toString(){
		StringBuilder b = new StringBuilder();
		b.append(Arrays.toString(dims));
		b.append("\n");
		b.append(Arrays.toString(get()));
		return b.toString();
	}
	
	public void finalize(){
		free(address);
	}
	
	private native long init(float[] data, int[] dims);
	
	private native void free(long address);
	
	private native int[] dims(long src);
	
	private native void reshape(long src, int... d);

	private native float get(long src, int... d);

	private native float[] get(long src);
	
	private native void set(long src, float v, int... d);

	private native void set(long src, float[] data);
	
	private native void fill(long src, float v);

	private native void rand(long src);

	private native void randn(long src);
	
	private native void bernoulli(long src, float p);

	private native long copyInto(long src, long other);

	private native long narrow(long src, int dim, int index, int size);

	private native long select(long src, int dim, int index);

	private native long transpose(long src, int d1, int d2);

	private native long diag(long src, long dst);
	
	private native boolean equals(long src, long other, float threshold);
}
