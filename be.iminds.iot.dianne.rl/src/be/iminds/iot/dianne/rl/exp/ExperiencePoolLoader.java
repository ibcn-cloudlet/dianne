package be.iminds.iot.dianne.rl.exp;

import java.util.Dictionary;
import java.util.Hashtable;

import org.osgi.framework.BundleContext;
import org.osgi.service.cm.Configuration;
import org.osgi.service.cm.ConfigurationAdmin;
import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;

@Component(immediate=true)
public class ExperiencePoolLoader {

	private ConfigurationAdmin configAdmin;
	
	@Activate
	public void activate(BundleContext context) throws Exception {
		
		String factoryPid = FileExperiencePool.class.getName();
		
		// TODO scan a directory / config file for this? for now hard coded for our single use case
		Dictionary<String, Object> properties = new Hashtable<>();
		properties.put("name", "Pong");
		properties.put("dir", "../tools/exp/pong");
		properties.put("stateSize", 6);
		properties.put("actionSize", 3);
		properties.put("labels", new String[]{"Left","Hold","Right"});
		
		String instancePid = configAdmin.createFactoryConfiguration(factoryPid, null).getPid();
		configAdmin.getConfiguration(instancePid).update(properties);
	}
	
	@Reference
	public void setConfigurationAdmin(ConfigurationAdmin ca){
		this.configAdmin = ca;
	}
}
