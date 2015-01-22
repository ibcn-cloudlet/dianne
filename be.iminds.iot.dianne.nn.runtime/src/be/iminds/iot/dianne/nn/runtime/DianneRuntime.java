package be.iminds.iot.dianne.nn.runtime;

import java.util.Collections;
import java.util.Dictionary;
import java.util.HashMap;
import java.util.Map;

import org.osgi.framework.BundleContext;
import org.osgi.framework.ServiceRegistration;
import org.osgi.service.cm.ConfigurationException;
import org.osgi.service.cm.ManagedServiceFactory;
import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Deactivate;
import org.osgi.service.component.annotations.Reference;

import be.iminds.iot.dianne.nn.module.Module;
import be.iminds.iot.dianne.nn.module.factory.ModuleFactory;
import be.iminds.iot.dianne.tensor.TensorFactory;

@Component(immediate=true, property={"service.pid=be.iminds.iot.dianne.nn.module"})
public class DianneRuntime implements ManagedServiceFactory {

	private BundleContext context;
	
	// TODO support multiple factories in the future?!
	private TensorFactory tFactory;
	private ModuleFactory mFactory;
	
	private Map<String, ServiceRegistration> modules = Collections.synchronizedMap(new HashMap<String, ServiceRegistration>());
	
	@Override
	public String getName() {
		return "be.iminds.iot.dianne.nn.module";
	}

	@Activate
	public void activate(BundleContext context){
		this.context = context;
	}
	
	@Deactivate
	public void deactivate(){
		for(ServiceRegistration reg : modules.values()){
			reg.unregister();
		}
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
		// Create and register module
		try {
			Module module = this.mFactory.createModule(tFactory, properties);
			
			ServiceRegistration reg = context.registerService(Module.class.getName(), module, properties);
			this.modules.put(pid, reg);
			
			System.out.println("Registered module "+module.getClass().getName()+" "+module.getId());
		} catch(InstantiationException e){
			System.err.println("Could not instantiate module");
			e.printStackTrace();
		}
	}

	@Override
	public void deleted(String pid) {
		ServiceRegistration reg = modules.get(pid);
		if(reg!=null){
			reg.unregister();
		}
	}

}
