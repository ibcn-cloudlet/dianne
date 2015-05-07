package be.iminds.iot.dianne.tensor.impl.nd4j;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.osgi.service.component.annotations.Component;

import be.iminds.iot.dianne.tensor.TensorFactory;
import be.iminds.iot.dianne.tensor.TensorMath;

@Component(property={"aiolos.export=false"})
public class ND4JTensorFactory implements TensorFactory<ND4JTensor>{

	private final ND4JTensorMath math = new ND4JTensorMath();
	
	public ND4JTensorFactory() {
		Nd4j.factory().setOrder('c');
	}
	
	@Override
	public ND4JTensor createTensor(int... d) {
		INDArray nd = Nd4j.create(d);
		return new ND4JTensor(nd);
	}

	@Override
	public ND4JTensor createTensor(float[] data, int... d) {
		INDArray nd = Nd4j.create(data, d);
		return new ND4JTensor(nd);
	}

	@Override
	public TensorMath<ND4JTensor> getTensorMath() {
		return math;
	}

}
