package be.iminds.iot.dianne.tensor;

public interface TensorMath<T extends Tensor<T>> {

	public T add(T result, T t1, T t2);
	
	public T mul(T result, T t1, T t2);
	
}
