package be.iminds.iot.dianne.tensor;

/**
 * Provides all supported Tensor operations. Each operation where a tensor is returned,
 * also has the argument res, in which one could provide a tensor in which the result
 * will be put and returned. This in order to save memory allocations. When res is null 
 * a new Tensor object will be created.
 * 
 * @author tverbele
 *
 */
public interface TensorMath<T extends Tensor<T>> {

	/**
	 * Add the given value to all elements in the T.
	 */
	public T add(T res, final T tensor, final float value);
	
	/**
	 * Add tensor1 to tensor2 and put result into res. 
	 * The number of elements must match, but sizes do not matter.
	 */
	public T add(T res, final T tensor1, final T tensor2);

	/**
	 * Multiply elements of tensor2 by the scalar value and add it to tensor1. 
	 * The number of elements must match, but sizes do not matter.
	 */
	public T add(T res, final T tensor1, final float value, final T tensor2);
	
	/**
	 * Multiply all elements in the T by the given value.
	 */
	public T mul(T res, final T tensor1, final float value);
	
	/**
	 * Element-wise multiplication of tensor1 by tensor2. 
	 * The number of elements must match, but sizes do not matter.
	 */
	public T cmul(T res, final T tensor1, final T tensor2);
	
	/**
	 * Divide all elements in the T by the given value.
	 */
	public T div(T res, final T tensor1, final float value);
	
	/**
	 * Element-wise division of tensor1 by tensor2. 
	 * The number of elements must match, but sizes do not matter.
	 */
	public T cdiv(T res, final T tensor1, final T tensor2);
	
	/**
	 * Performs the dot product between vec1 and vec2. 
	 * The number of elements must match: both Ts are seen as a 1D vector.
	 */
	public float dot(final T vec1, final T vec2);
	
	/**
	 * Matrix vector product of mat and vec. 
	 * Sizes must respect the matrix-multiplication operation: 
	 * if mat is a n x m matrix, vec must be vector of size m and res must be a vector of size n.
	 */
	public T mv(T res, final T mat, final T vec);
	
	/**
	 * Matrix matrix product of matensor1 and matensor2. If matensor1 is a n x m matrix, matensor2 a m x p matrix, 
	 * res must be a n x p matrix.
	 */
	public T mm(T res, final T mat1, final T mat2);
	
	/**
	 * Performs a matrix-vector multiplication between mat (2D tensor) and vec (1D tensor) 
	 * and add it to vec1. In other words, res = vec1 + mat*vec2
	 */
	public T addmv(T res, final T vec1, final T mat, final T vec2);

	/**
	 * Performs a matrix-vector multiplication between matensor1 (2D tensor) and matensor2 (2D tensor) 
	 * and add it to mat. In other words, res = mat + matensor1*matensor2
	 */
	public T addmm(T res, final T mat, final T mat1, final T mat2);

	/**
	 * Put the tanh of each element into res
	 */
	public T tanh(T res, final T tensor);
	
	/**
	 * Calculates for each element (1-x^2)
	 */
	public T dtanh(T res, final T tensor);
	
	/**
	 * Put the sigmoid of each element into res
	 */
	public T sigmoid(T res, final T tensor);
	
	/**
	 * Calculates for each element x*(1-x)
	 */
	public T dsigmoid(T res, final T tensor);
}
