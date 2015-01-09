package be.iminds.iot.dianne.tensor;

import java.lang.reflect.InvocationTargetException;

import be.iminds.iot.dianne.tensor.impl.java.JavaTensor;
import be.iminds.iot.dianne.tensor.impl.java.JavaTensorMath;

/**
 * Factory interface to create Tensors and get a suitable TensorMath object for these Tensors.
 * Within a single runtime, every Tensor should be created via the TensorFactory and only the
 * matching TensorMath object should be used. 
 * 
 * @author tverbele
 *
 */
public class TensorFactory<T extends Tensor<T>> {
	
	private Class<T> type;
	private final TensorMath<T> math;
	
	public <R extends TensorMath<T>> TensorFactory(Class<T> tensorType, Class<R> mathType) {
		super();
		this.type = tensorType;
		try {
			this.math = mathType.getConstructor(TensorFactory.class).newInstance(this);
		} catch (InstantiationException | IllegalAccessException
				| IllegalArgumentException | InvocationTargetException
				| NoSuchMethodException | SecurityException e) {
			throw new IllegalArgumentException(e);
		}
	}

	public T createTensor(final int ... d){
		try {
			return type.getConstructor(int[].class).newInstance(d);
		} catch (InstantiationException | IllegalAccessException
				| IllegalArgumentException | InvocationTargetException
				| NoSuchMethodException | SecurityException e) {
			throw new IllegalArgumentException(e);
		}
	}
	
	public TensorMath<T> getTensorMath(){
		return math;
	}
	
	public static TensorFactory<?> getFactory(TensorType t){
		switch(t){
			case JAVA:
				return new TensorFactory<JavaTensor>(JavaTensor.class, JavaTensorMath.class);
			default:
				throw new IllegalArgumentException("Invalid Tensor type!");
		}
	}
	
	public enum TensorType{
		JAVA
	}
	
}
