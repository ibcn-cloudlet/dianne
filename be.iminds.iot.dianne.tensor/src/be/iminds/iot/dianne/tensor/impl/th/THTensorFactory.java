package be.iminds.iot.dianne.tensor.impl.th;

import org.osgi.service.component.annotations.Component;

import be.iminds.iot.dianne.tensor.TensorFactory;
import be.iminds.iot.dianne.tensor.TensorMath;

@Component(property={"aiolos.export=false"})
public class THTensorFactory implements TensorFactory<THTensor>{

	@Override
	public THTensor createTensor(int... d) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public THTensor createTensor(float[] data, int... d) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public TensorMath<THTensor> getTensorMath() {
		// TODO Auto-generated method stub
		return null;
	}

}
