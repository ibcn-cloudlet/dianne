package be.iminds.iot.dianne.rl.learn.command;

import java.util.HashMap;
import java.util.Map;

import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;

import be.iminds.iot.dianne.api.nn.learn.Learner;

/**
 * Separate component for learn commands ... should be moved to the command bundle later on
 */
@Component(
		service=Object.class,
		property={"osgi.command.scope=dianne",
				  "osgi.command.function=learn"},
		immediate=true)
public class DianneRLLearnCommands {

	private Learner learner;
	
	public void learn(String nnName, String dataset, String tag){
		try {
			Map<String, String> config = new HashMap<String, String>();
			if(tag!=null){
				config.put("tag", tag);
			}
			learner.learn(nnName, dataset, config);
		} catch(Exception e){
			e.printStackTrace();
		}
	}

	public void learn(String nnName, String dataset){
		learn(nnName, dataset, null);
	}
	
	@Reference
	public void setLearner(Learner l){
		this.learner = l;
	}
}
