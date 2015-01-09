package be.iminds.iot.dianne.tensor;

/**
 * Represents an n-dimensional tensor
 * 
 * A Tensor should implement the equals method to check if 
 * two tensors are equal.
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
	 * get a value of the tensor using a flat index (seeing it as a 1d vector)
	 */
	public float get(final int i);
	
	/**
	 * set a value of the tensor
	 * @param v the new value
	 * @param d the indices of the element to set
	 */
	public void set(final float v, final int... d);

	/**
	 * set a value of the tensor using a flat index (seeing it as a 1d vector)
	 * @param v the new value
	 * @param i the indices of the element to set
	 */
	public void set(final float v, final int i);

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
	public boolean sameDim(Tensor other);
}
