package be.iminds.iot.dianne.tensor;

import org.osgi.framework.BundleContext;
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
	public void activate(BundleContext context){
		int device = -1;
		if(context != null){
			String d = context.getProperty("be.iminds.iot.dianne.tensor.device");
			if(d != null){
				device = Integer.parseInt(d);
			}
		}
		
		init(device);
	}
	
	@Deactivate()
	public void deactivate(){
		cleanup();
	}
	
	// set GPU device id in case of multiple GPUs on machine!
	private native void init(int device);
	
	private native void cleanup();
	
}
