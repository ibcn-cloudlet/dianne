package be.iminds.iot.dianne.rl.command;

import java.util.HashMap;
import java.util.Map;

import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;

import be.iminds.iot.dianne.api.nn.learn.Learner;

/**
 * Separate component for rl commands ... should be moved to the command bundle later on
 */
@Component(
		service=Object.class,
		property={"osgi.command.scope=dianne",
				  "osgi.command.function=learn",
				  "osgi.command.function=stopLearn"},
		immediate=true)
public class DianneRLLearnCommands {

	private Learner learner;

	public void learn(String nnName, String dataset, String ... properties){
		try {
			learner.learn(nnName, dataset, createConfig(properties));
		} catch(Exception e){
			e.printStackTrace();
		}
	}

	public void stopLearn(){
		this.learner.stop();
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
	public void setLearner(Learner l){
		this.learner = l;
	}
}
