package be.iminds.iot.dianne.coordinator;

import java.util.HashMap;
import java.util.Map;
import java.util.UUID;

import be.iminds.iot.dianne.api.coordinator.AgentResult;
import be.iminds.iot.dianne.api.coordinator.Job.Type;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkDTO;
import be.iminds.iot.dianne.api.rl.agent.Agent;
import be.iminds.iot.dianne.api.rl.agent.AgentProgress;

public class ActJob extends AbstractJob<AgentResult> {

	private Map<UUID, Agent> agents = new HashMap<>();
	
	public ActJob(DianneCoordinatorImpl coord,
			NeuralNetworkDTO nn,
			String d,
			Map<String, String> c){
		super(coord, Type.ACT, nn, d, c);
	}
	
	@Override
	public void execute() throws Exception {
		
		String environment = config.get("environment");
		
		// start acting
		System.out.println("Start Act Job");
		System.out.println("===============");
		System.out.println("* nn: "+nn.name);
		System.out.println("* pool: "+dataset);
		System.out.println("* environment: "+environment);
		System.out.println("---");
		
		// start learning on each learner
		for(UUID target : targets){
			Agent agent = coordinator.agents.get(target);
			agents.put(target, agent);
			agent.act(dataset, config, nnis.get(target), environment);
		}
		
		
		// TODO call when stopped?
		// done((Void)null);
	}

	@Override
	public AgentResult getProgress() {
		Map<UUID, AgentProgress> results = new HashMap<>();
		for(Agent a : agents.values()){
			results.put(a.getAgentId(), a.getProgress());
		}
		return new AgentResult(results);
	}

}
