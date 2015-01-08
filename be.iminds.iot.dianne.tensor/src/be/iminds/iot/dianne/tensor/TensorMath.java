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
public interface TensorMath {

	/**
	 * Add the given value to all elements in the tensor.
	 */
	public Tensor add(final Tensor res, final Tensor tensor, final float value);
	
	/**
	 * Add tensor1 to tensor2 and put result into res. 
	 * The number of elements must match, but sizes do not matter.
	 */
	public Tensor add(final Tensor res, final Tensor tensor1, final Tensor tensor2);

	/**
	 * Multiply elements of tensor2 by the scalar value and add it to tensor1. 
	 * The number of elements must match, but sizes do not matter.
	 */
	public Tensor add(final Tensor res, final Tensor tensor1, final float value, final Tensor tensor2);
	
	/**
	 * Multiply all elements in the tensor by the given value.
	 */
	public Tensor mul(final Tensor res, final Tensor tensor1, final float value);
	
	/**
	 * Element-wise multiplication of tensor1 by tensor2. 
	 * The number of elements must match, but sizes do not matter.
	 */
	public Tensor cmul(final Tensor res, final Tensor tensor1, final Tensor tensor2);
	
	/**
	 * Divide all elements in the tensor by the given value.
	 */
	public Tensor div(final Tensor res, final Tensor tensor1, final float value);
	
	/**
	 * Element-wise division of tensor1 by tensor2. 
	 * The number of elements must match, but sizes do not matter.
	 */
	public Tensor cdiv(final Tensor res, final Tensor tensor1, final Tensor tensor2);
	
	/**
	 * Performs the dot product between vec1 and vec2. 
	 * The number of elements must match: both tensors are seen as a 1D vector.
	 */
	public float dot(final Tensor vec1, final Tensor vec2);
	
	/**
	 * Matrix vector product of mat and vec. 
	 * Sizes must respect the matrix-multiplication operation: 
	 * if mat is a n x m matrix, vec must be vector of size m and res must be a vector of size n.
	 */
	public Tensor mv(final Tensor res, final Tensor mat, final Tensor vec);
	
	/**
	 * Matrix matrix product of mat1 and mat2. If mat1 is a n x m matrix, mat2 a m x p matrix, 
	 * res must be a n x p matrix.
	 */
	public Tensor mm(final Tensor res, final Tensor mat1, final Tensor mat2);
	
}
