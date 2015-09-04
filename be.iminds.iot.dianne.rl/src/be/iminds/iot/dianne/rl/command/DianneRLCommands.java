package be.iminds.iot.dianne.rl.command;

import java.io.File;
import java.io.FileWriter;
import java.io.PrintStream;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;
import org.osgi.service.component.annotations.ReferenceCardinality;
import org.osgi.service.component.annotations.ReferencePolicy;

import be.iminds.iot.dianne.api.nn.learn.Learner;
import be.iminds.iot.dianne.api.rl.Agent;
import be.iminds.iot.dianne.api.rl.ExperiencePool;

/**
 * Separate component for rl commands ... should be moved to the command bundle later on
 * 
 * Combo of Agent + Learner
 */
@Component(
		service=Object.class,
		property={"osgi.command.scope=dianne",
				  "osgi.command.function=rl",
				  "osgi.command.function=dump"},
		immediate=true)
public class DianneRLCommands {

	private Agent agent;
	private Learner learner;
	private Map<String, ExperiencePool> pools = new HashMap<String, ExperiencePool>();

	
	public void rl(String nnName, String environment, String experiencePool, String... properties) throws Exception {
		Map<String, String> agentConfig = DianneRLAgentCommands.createAgentConfig(properties);
		Map<String, String> learnConfig = DianneRLLearnCommands.createLearnConfig(properties);
		if(!agentConfig.containsKey("tag")){
			agentConfig.put("tag", "run");
			learnConfig.put("tag", "run");
		}
		agent.act(nnName, environment, experiencePool, agentConfig);
		
		learner.learn(nnName, experiencePool, learnConfig);
	}
	
	public void dump(String name) throws Exception{
		dump(name, null, -1);
	}
	
	public void dump(String name, int count) throws Exception{
		dump(name, null, count);
	}
	
	public void dump(String name, String file) throws Exception{
		dump(name, file, -1);
	}
	
	public void dump(String name, String file, int count) throws Exception {
		ExperiencePool p = pools.get(name);
		PrintStream writer; 
		if(file!=null){
			writer = new PrintStream(new File(file));
		} else {
			writer = System.out;
		}
		int start = 0;
		if(count > 0){
			start = p.size() - count;
			start = start > 0 ? start : 0;
		}
		for(int i=start;i<p.size();i++){
			writer.println(i+"\t"+Arrays.toString(p.getState(i).get())+"\t"+Arrays.toString(p.getAction(i).get())+"\t"+p.getReward(i)+"\t"+Arrays.toString(p.getNextState(i).get()));
		}
	}
	
	@Reference
	public void setAgent(Agent agent){
		this.agent = agent;
	}
	
	@Reference
	public void setLearner(Learner l){
		this.learner = l;
	}
	
	@Reference(cardinality = ReferenceCardinality.MULTIPLE, policy = ReferencePolicy.DYNAMIC)
	public void addExperiencePool(ExperiencePool pool, Map<String, Object> properties) {
		String name = (String) properties.get("name");
		this.pools.put(name, pool);
	}

	public void removeExperiencePool(ExperiencePool pool, Map<String, Object> properties) {
		String name = (String) properties.get("name");
		this.pools.remove(name);
	}
}
