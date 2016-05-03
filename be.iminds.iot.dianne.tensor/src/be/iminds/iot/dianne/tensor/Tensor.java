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

/**
 * Represents an n-dimensional tensor
 * 
 * A Tensor should implement the equals method to check if 
 * two tensors are equal.
 * 
 * 
 * @author tverbele
 *
 */
public interface Tensor<T extends Tensor<T>> {

	/**
	 * @return the number of dimensions of this tensor
	 */
	int dim();

	/**
	 * @return the dimensions of this tensor
	 */
	int[] dims();
	
	/**
	 * @return the total size of the tensor
	 */
	int size();
	
	/**
	 * the size of the d'th dimension
	 * @param d the dimension to query the size
	 * @return the size of the dimension
	 */
	int size(final int d);
	
	/**
	 * reshape the dimensions of this tensor, the underlying data remains the same
	 */
	void reshape(final int... d);
	
	/** 
	 * get a value of the tensor
	 * @param d indices of the element
	 * @return the element specified by the index
	 */
	float get(final int... d);
	
	/**
	 * get (a copy of) the raw data for this tensor, this way that the tensor 
	 * can be reconstructed with the createTensor(data, dims) factory method
	 */
	float[] get();
	
	/**
	 * set a value of the tensor
	 * @param v the new value
	 * @param d the indices of the element to set
	 */
	void set(final float v, final int... d);
	
	/**
	 *  copy a complete array of raw data into this tensor
	 */
	void set(final float[] data);
	
	/**
	 * fill with fixed value
	 * @param v the new value
	 */
	void fill(final float v);

	/**
	 * fill with random values uniformely distributed between 0 and 1
	 */
	void rand();

	/**
	 * fill with random values Gaussian ("normally") distributed with mean 0.0 and standard deviation 1.0
	 */
	void randn();
	
	/**
	 * fill with 0 or 1 sampled using Bernoulli distribution with 0 <= p <= 1
	 */
	void bernoulli(float p);
	
	/**
	 * check if other tensor has same dimensions
	 */
	boolean sameDim(final Tensor<?> other);
	
	/**
	 * check if other tensor has these dimensions
	 */
	boolean hasDim(final int... dims);
	
	/**
	 * clone this tensor into other tensor, create new one if null or different number of elements
	 * @param other the tensor to clone into
	 * @return the cloned tensor
	 */
	T copyInto(final T other);
	
	/**
	 * Return a subtensor narrowing dimension dim from index to index+size-1
	 */
	T narrow(final int dim, final int index, final int size);
	
	/**
	 * Return a subtensor narrowing according to the ranges array. This is interpreted
	 * as narrowing dimension 1 from ranges[0] with size ranges[1], narrowing dimension 2 from
	 * ranges[2] with size ranges[3], etc.
	 */
	T narrow(final int... ranges);
	
	/**
	 * Return a slice at the given index in dimension dim, dimension dim will be removed
	 */
	T select(final int dim, final int index);
	
	/**
	 * calculate the transpose of the tensor
	 */
	T transpose(T res, final int d1, final int d2);
	
	/**
	 * return the diag vec of the tensor
	 */
	T diag(T res);
	
	/**
	 * return whether two tensors are equal (note: they have to be the same type to be equal!)
	 * @param other object to compare to
	 * @return true if the other object represents an equal tensor
	 */
	boolean equals(T other);
	
	/**
	 * equals with threshold (note: they have to be the same type to be equal!)
	 * @param other object to compare to
	 * @return true if the other object represents an equal tensor with values within threshold range
	 */
	boolean equals(T other, float threshold);
}
