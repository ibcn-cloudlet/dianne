package be.iminds.iot.dianne.nn.runtime;

import java.util.Dictionary;

import org.osgi.framework.BundleContext;
import org.osgi.service.cm.ConfigurationException;
import org.osgi.service.cm.ManagedServiceFactory;
import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Deactivate;
import org.osgi.service.component.annotations.Reference;

import be.iminds.iot.dianne.nn.module.factory.ModuleFactory;
import be.iminds.iot.dianne.tensor.TensorFactory;

@Component(immediate=true, property={"service.pid=be.iminds.iot.dianne.nn.module"})
public class DianneRuntime implements ManagedServiceFactory {

	private BundleContext context;
	
	// TODO support multiple factories in the future?!
	private TensorFactory tFactory;
	private ModuleFactory mFactory;
	
	@Override
	public String getName() {
		return "be.iminds.iot.dianne.nn.module";
	}

	@Activate
	public void activate(BundleContext context){
		this.context = context;
		System.out.println("ACTIVATED!");
	}
	
	@Deactivate
	public void deactivate(){
		// Tear down all modules!
	}
	
	@Reference
	public void setTensorFactory(TensorFactory factory){
		this.tFactory = factory;
	}
	
	@Reference
	public void setModuleFactory(ModuleFactory factory){
		this.mFactory = factory;
	}
	
	@Override
	public void updated(String pid, Dictionary<String, ?> properties)
			throws ConfigurationException {
		System.out.println("UPDATED "+pid);
		
		// Create and register module
		
	}

	@Override
	public void deleted(String pid) {
		System.out.println("DELETED "+pid);
		
		// Delete and unregister module
	}

}
