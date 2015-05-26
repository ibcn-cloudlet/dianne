package be.iminds.iot.dianne.tensor.impl.th;

import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Deactivate;

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
    
    public THTensorFactory() {
    	init();
	}
    
	@Activate
	public void activate(){
		// call init in constructor to also work in JUnit testing
		//init();
	}
	
	@Deactivate
	public void deactivate(){
		cleanup();
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

	private native void init();
	
	private native void cleanup();
}
