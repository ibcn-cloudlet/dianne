package be.iminds.iot.dianne.rl.command;

import java.util.HashMap;
import java.util.Map;

import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;

import be.iminds.iot.dianne.api.nn.learn.Learner;
import be.iminds.iot.dianne.api.rl.Agent;

/**
 * Separate component for rl commands ... should be moved to the command bundle later on
 */
@Component(
		service=Object.class,
		property={"osgi.command.scope=dianne",
				  "osgi.command.function=act",
				  "osgi.command.function=stopAct",
				  "osgi.command.function=learn",
				  "osgi.command.function=stopLearn"},
		immediate=true)
public class DianneRLCommands {

	private Agent agent;
	private Learner learner;
	
	public void act(String nnName, String environment){
		act(nnName, environment, null);
	}
	
	public void act(String nnName, String environment, String experiencePool){
		try {
			agent.act(nnName, environment, experiencePool, new HashMap<String, String>());
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	public void stopAct(){
		this.agent.stop();
	}
	
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
	public void setAgent(Agent agent){
		this.agent = agent;
	}
	
	@Reference
	public void setLearner(Learner l){
		this.learner = l;
	}
}
