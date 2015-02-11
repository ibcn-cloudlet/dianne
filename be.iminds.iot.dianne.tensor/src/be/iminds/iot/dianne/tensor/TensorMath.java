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
	 * Subract the given value of all elements in the T.
	 */
	public T sub(T res, final T tensor, final float value);
	
	/**
	 * Subtract tensor2 from tensor1 and put result into res. 
	 * The number of elements must match, but sizes do not matter.
	 */
	public T sub(T res, final T tensor1, final T tensor2);
	
	/**
	 * Multiply elements of tensor2 by the scalar value and subtract it from tensor1. 
	 * The number of elements must match, but sizes do not matter.
	 */
	public T sub(T res, final T tensor1, final float value, final T tensor2);
	
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
	 * Element-wise division res = t1/t2
	 */
	public T div(T res, final T tensor1, final T  tensor2);
	
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
	 * Performs the matrix product between vec1 and vec2
	 * @param res placeholder
	 * @param vec1 vector of size m
	 * @param vec2 vector of size n
	 * @return resulting matrix of size mxn
	 */
	public T vv(T res, final T vec1, final T vec2);
	
	/**
	 * Matrix vector product of mat and vec. 
	 * Sizes must respect the matrix-multiplication operation: 
	 * if mat is a n x m matrix, vec must be vector of size m and res must be a vector of size n.
	 */
	public T mv(T res, final T mat, final T vec);
	
	/**
	 * Matrix vector product of transposed mat and vec. 
	 * Sizes must respect the matrix-multiplication operation: 
	 * if mat is a m x n matrix, vec must be vector of size m and res must be a vector of size n.
	 */
	public T tmv(T res, final T mat, final T vec);
	
	/**
	 * Matrix matrix product of matensor1 and matensor2. If matensor1 is a n x m matrix, matensor2 a m x p matrix, 
	 * res must be a n x p matrix.
	 */
	public T mm(T res, final T mat1, final T mat2);
	
	/**
	 * Performs the matrix product between vec1 and vec2 and adds this to mat
	 * @param res placeholder
	 * @param mat mxn matrix to add to result
	 * @param vec1 vector of size m
	 * @param vec2 vector of size n
	 * @return resulting matrix of size mxn
	 */
	public T addvv(T res, final T mat, final T vec1, final T vec2);
	
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
	 * Calculates element-wise exp function
	 */
	public T exp(T res, final T tensor);

	/**
	 * Calculates element-wise log function
	 */
	public T log(T res, final T tensor);
	
	/**
	 * Calculates element-wise tanh function
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
	
	/**
	 * All elements smaller than thresh are set to val
	 */
	public T thresh(T res, final T tensor, final float thresh, final float val);
	
	/**
	 * All elements smaller than thresh are set to 0, 1 otherwise
	 */
	public T dthresh(T res, final T tensor, final float thresh);
	
	/**
	 * Calculates element-wise softmax function
	 */
	public T softmax(T res, final T tensor);
	
	/**
	 * Return the sum of all elements
	 */
	public float sum(final T tensor);
	
	/**
	 * Return the max of all elements
	 */
	public float max(final T tensor);
	
	/**
	 * Return the min of all elements
	 */
	public float min(final T tensor);
	
	/**
	 * Return the mean of all elements
	 */
	public float mean(final T tensor);
	
	/**
	 * Return index of the max element (treats T as 1 dim vector)
	 */
	public int argmax(final T tensor);
	
	/**
	 * Return index of the min element (treats T as 1 dim vector)
	 */
	public int argmin(final T tensor);
	
	/**
	 * Calculate 2D convolution mat1 * mat2
	 */
	public T convolution2D(T res, final T mat1, final T mat2);
	
	/**
	 * Max pooling of tensor mat and put result in res
	 */
	public T maxpool2D(T res, final T mat, final int w, final int h);
	
	/**
	 * Backward of max pooling: calculate each max index in wxh block of mat1, and 
	 * put value of subsampled mat2 into res on that position 
	 */
	public T dmaxpool2D(T res, final T mat2, final T mat1, final int w, final int h);
}
