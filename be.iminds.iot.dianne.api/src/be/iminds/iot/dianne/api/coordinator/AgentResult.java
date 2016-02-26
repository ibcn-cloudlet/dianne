package be.iminds.iot.dianne.api.coordinator;

import java.util.Map;
import java.util.UUID;

import be.iminds.iot.dianne.api.rl.agent.AgentProgress;

public class AgentResult {
	
	public Map<UUID, AgentProgress> results;
	
	public AgentResult(Map<UUID, AgentProgress> results){
		this.results = results;
	}
}

