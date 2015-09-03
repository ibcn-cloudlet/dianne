package be.iminds.iot.dianne.rl.command;

import java.util.Map;

import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;

import be.iminds.iot.dianne.api.nn.learn.Learner;
import be.iminds.iot.dianne.api.rl.Agent;

/**
 * Separate component for rl commands ... should be moved to the command bundle later on
 * 
 * Combo of Agent + Learner
 */
@Component(
		service=Object.class,
		property={"osgi.command.scope=dianne",
				  "osgi.command.function=rl"},
		immediate=true)
public class DianneRLCommands {

	private Agent agent;
	private Learner learner;
	
	public void rl(String nnName, String environment, String experiencePool, String... properties) throws Exception {
		Map<String, String> agentConfig = DianneRLAgentCommands.createAgentConfig(properties);
		Map<String, String> learnConfig = DianneRLLearnCommands.createLearnConfig(properties);
		if(!agentConfig.containsKey("tag")){
			agentConfig.put("tag", "run");
			learnConfig.put("tag", "run");
		}
		agent.act(nnName, environment, experiencePool, agentConfig);
		
		// sleep a while before starting training
		Thread.sleep(10000);
		
		learner.learn(nnName, experiencePool, learnConfig);
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
