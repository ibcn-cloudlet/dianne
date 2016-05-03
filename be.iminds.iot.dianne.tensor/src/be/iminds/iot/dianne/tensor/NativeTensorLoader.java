package be.iminds.iot.dianne.tensor;

import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Deactivate;

@Component(immediate=true)
public class NativeTensorLoader {

	static {
		try {
		    System.loadLibrary("Tensor");
		} catch (final UnsatisfiedLinkError e) {
		    System.err.println("Native code library Tensor failed to load. \n"+ e);
		    throw e;
		}
	}
	
	@Activate
	void activate(){
		init();
	}
	
	@Deactivate()
	void deactivate(){
		cleanup();
	}
	
	private native void init();
	
	private native void cleanup();
	
}
