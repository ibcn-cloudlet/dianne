package be.iminds.iot.dianne.nn.runtime;

import java.util.Dictionary;

import org.osgi.framework.BundleContext;
import org.osgi.service.cm.ConfigurationException;
import org.osgi.service.cm.ManagedServiceFactory;
import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Deactivate;

@Component(immediate=true, property={"service.pid=be.iminds.iot.dianne.nn.module"})
public class DianneRuntime implements ManagedServiceFactory {

	private BundleContext context;
	
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
