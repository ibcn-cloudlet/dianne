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
	public int dim();

	/**
	 * @return the dimensions of this tensor
	 */
	public int[] dims();
	
	/**
	 * @return the total size of the tensor
	 */
	public int size();
	
	/**
	 * the size of the d'th dimension
	 * @param d the dimension to query the size
	 * @return the size of the dimension
	 */
	public int size(final int d);
	
	/** 
	 * get a value of the tensor
	 * @param d indices of the element
	 * @return the element specified by the index
	 */
	public float get(final int... d);
	
	/**
	 * set a value of the tensor
	 * @param v the new value
	 * @param d the indices of the element to set
	 */
	public void set(final float v, final int... d);

	/**
	 * fill with fixed value
	 * @param v the new value
	 */
	public void fill(final float v);

	/**
	 * fill with random values
	 */
	public void rand();
	
	/**
	 * check if other tensor has same dimensions
	 */
	public boolean sameDim(Tensor<?> other);
	
	/**
	 * clone this tensor into other tensor, create new one if null or different number of elements
	 * @param other the tensor to clone into
	 * @return the cloned tensor
	 */
	public T clone(final T other);
	
	/**
	 * Return a subtensor narrowing dimension dim from index to index+size-1
	 */
	public T narrow(final int dim, final int index, final int size);
	
	/**
	 * Return a subtensor narrowing according to the ranges array. This is interpreted
	 * as narrowing dimension 1 from ranges[0] with size ranges[1], narrowing dimension 2 from
	 * ranges[2] with size ranges[3], etc.
	 */
	public T narrow(final int... ranges);
	
	/**
	 * Return a slice at the given index in dimension dim
	 */
	public T select(final int dim, final int index);
	
	/**
	 * calculate the transpose of the tensor
	 */
	public T transpose(T res, final int d1, final int d2);
}
