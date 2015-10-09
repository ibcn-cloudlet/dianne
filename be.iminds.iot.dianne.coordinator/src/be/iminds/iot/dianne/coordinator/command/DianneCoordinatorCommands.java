package be.iminds.iot.dianne.coordinator.command;

import java.util.HashMap;
import java.util.Map;

import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;

import be.iminds.iot.dianne.api.coordinator.DianneCoordinator;

/**
 * Separate component for learn commands ... should be moved to the command bundle later on
 */
@Component(
		service=Object.class,
		property={"osgi.command.scope=dianne",
				  "osgi.command.function=jobLearn"},
		immediate=true)
public class DianneCoordinatorCommands {

	private DianneCoordinator coordinator;
	
	public void jobLearn(String nnName, String dataset, String... properties){
		try {
			Map<String, String> config = createConfig(properties);
			
			coordinator.learn(nnName, dataset, config).then(p -> {System.out.println("Job done!!! "+ p.getValue()); return null;});
		} catch(Exception e){
			e.printStackTrace();
		}
	}
	
	private Map<String, String> createConfig(String[] properties){
		Map<String, String> config = new HashMap<String, String>();
		for(String property : properties){
			String[] p = property.split("=");
			if(p.length==2){
				config.put(p[0].trim(), p[1].trim());
			}
		}
		
		return config;
	}
	
	
	@Reference
	void setDianneCoordinator(DianneCoordinator c){
		this.coordinator = c;
	}
	
}
