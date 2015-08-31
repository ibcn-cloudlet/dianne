package be.iminds.iot.dianne.rl.command;

import java.util.HashMap;

import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;

import be.iminds.iot.dianne.api.rl.Agent;

/**
 * Separate component for rl commands ... should be moved to the command bundle later on
 */
@Component(
		service=Object.class,
		property={"osgi.command.scope=dianne",
				  "osgi.command.function=act",
				  "osgi.command.function=stopAct"},
		immediate=true)
public class DianneRLCommands {

	// TODO could be multiple agents?
	private Agent agent;
	
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
	
	@Reference
	public void setAgent(Agent agent){
		this.agent = agent;
	}
}
