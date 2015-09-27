package be.iminds.iot.dianne.rl.command;

import java.util.HashMap;
import java.util.Map;

import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;

import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkInstanceDTO;
import be.iminds.iot.dianne.api.nn.platform.DiannePlatform;
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
public class DianneRLAgentCommands {

	private DiannePlatform platform;
	
	private Agent agent;
	
	public void act(String nnName, String environment){
		act(nnName, environment, null);
	}
	
	public void act(String nnName, String environment, String experiencePool, String... properties){
		try {
			NeuralNetworkInstanceDTO nni = platform.deployNeuralNetwork(nnName);
			
			agent.act(nni, environment, experiencePool, createAgentConfig(properties));
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	public void stopAct(){
		agent.stop();
	}
	
	static Map<String, String> createAgentConfig(String[] properties){
		Map<String, String> config = new HashMap<String, String>();
		// defaults
		config.put("strategy", "greedy");
		config.put("epsilonMax", "1");
		config.put("epsilonMin", "0.1");
		config.put("epsilonDecay", "1e-6");

		for(String property : properties){
			String[] p = property.split("=");
			if(p.length==2){
				config.put(p[0].trim(), p[1].trim());
			}
		}
		
		return config;
	}
	
	@Reference
	void setAgent(Agent agent){
		this.agent = agent;
	}
	
	@Reference
	void setDiannePlatform(DiannePlatform p){
		this.platform = p;
	}
}
