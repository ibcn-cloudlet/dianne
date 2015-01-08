package be.iminds.iot.dianne.tensor;

public interface Tensor<T extends Tensor<T>> {

	/**
	 * @return the number of dimensions of this tensor
	 */
	public int dim();
	
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
	
}
