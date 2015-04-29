package be.iminds.iot.dianne.tensor.impl.th;

import org.osgi.service.component.annotations.Component;

import be.iminds.iot.dianne.tensor.TensorFactory;
import be.iminds.iot.dianne.tensor.TensorMath;

@Component(property={"aiolos.export=false"})
public class THTensorFactory implements TensorFactory<THTensor>{

    static {
		try {
		    System.loadLibrary("THTensor");
		} catch (final UnsatisfiedLinkError e) {
		    System.err.println("Native code library THTensor failed to load. \n"+ e);
		    throw e;
		}
    }
    
    private final THTensorMath math = new THTensorMath();
	
	@Override
	public THTensor createTensor(int... d) {
		return new THTensor(d);
	}

	@Override
	public THTensor createTensor(float[] data, int... d) {
		return new THTensor(data, d);
	}

	@Override
	public TensorMath<THTensor> getTensorMath() {
		return math;
	}

}
