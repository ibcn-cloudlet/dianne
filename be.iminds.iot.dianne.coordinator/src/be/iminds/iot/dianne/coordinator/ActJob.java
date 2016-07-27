/*******************************************************************************
 * DIANNE  - Framework for distributed artificial neural networks
 * Copyright (C) 2015  iMinds - IBCN - UGent
 *
 * This file is part of DIANNE.
 *
 * DIANNE is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * Contributors:
 *     Tim Verbelen, Steven Bohez
 *******************************************************************************/
package be.iminds.iot.dianne.coordinator;

import java.util.Arrays;
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
			String dataset,
			Map<String, String> config,
			NeuralNetworkDTO[] nns){
		super(coord, Type.ACT, dataset, config, nns);
	}
	
	@Override
	public void execute() throws Exception {
		
		String environment = config.get("environment");
		
		// start acting
		System.out.println("Start Act Job");
		System.out.println("===============");
		System.out.println("* nn: "+Arrays.toString(nnNames));
		System.out.println("* pool: "+dataset);
		System.out.println("* environment: "+environment);
		System.out.println("---");
		
		// start learning on each learner
		for(UUID target : targets){
			Agent agent = coordinator.agents.get(target);
			agents.put(target, agent);
			agent.act(environment, dataset, config, nnis.get(target));
		}
	}

	@Override
	public AgentResult getProgress() {
		Map<UUID, AgentProgress> results = new HashMap<>();
		for(Agent a : agents.values()){
			results.put(a.getAgentId(), a.getProgress());
		}
		return new AgentResult(results);
	}
	
	@Override
	public void cleanup() {
		for(Agent a : agents.values()){
			a.stop();
		}
	}

	@Override
	public void stop() throws Exception{
		if(started > 0){
			done(getProgress());
		} else {
			done(new Exception("Job "+this.jobId+" cancelled."));
		}
	}
}
