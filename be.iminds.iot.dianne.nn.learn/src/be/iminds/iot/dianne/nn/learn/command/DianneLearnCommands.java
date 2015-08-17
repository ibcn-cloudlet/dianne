package be.iminds.iot.dianne.nn.learn.command;

import java.util.HashMap;

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
public class DianneLearnCommands {

	private Learner learner;
	
	public void learn(String nnName, String dataset){
		try {
			learner.learn(nnName, dataset, new HashMap<String, String>());
		} catch(Exception e){
			e.printStackTrace();
		}
	}
	
	@Reference
	public void setLearner(Learner l){
		this.learner = l;
	}
}
