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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Dictionary;
import java.util.HashMap;
import java.util.Hashtable;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.stream.Collectors;

import org.osgi.framework.ServiceRegistration;

import be.iminds.iot.dianne.api.coordinator.AgentResult;
import be.iminds.iot.dianne.api.coordinator.Job.LearnCategory;
import be.iminds.iot.dianne.api.coordinator.Job.Type;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkDTO;
import be.iminds.iot.dianne.api.rl.agent.Agent;
import be.iminds.iot.dianne.api.rl.agent.AgentListener;
import be.iminds.iot.dianne.api.rl.agent.AgentProgress;

public class ActJob extends AbstractJob<AgentResult> implements AgentListener {

	private ServiceRegistration<AgentListener> reg;
	
	private long totalIterations = 0;
	private long totalSequences = 0;
	
	private long maxIterations = -1;
	private long maxSequences = -1;
	
	private Map<UUID, Agent> agents = new HashMap<>();
	
	private AgentResult result = new AgentResult();
	
	public ActJob(DianneCoordinatorImpl coord,
			String dataset,
			Map<String, String> config,
			NeuralNetworkDTO[] nns){
		super(coord, Type.ACT, dataset, config, nns);
		
		category = LearnCategory.RL;
	}
	
	@Override
	public void execute() throws JobFailedException {
		
		String environment = config.get("environment");
		
		if(config.containsKey("maxIterations")){
			maxIterations = Long.parseLong(config.get("maxIterations"));
		}
		
		if(config.containsKey("maxSequences")){
			maxSequences = Long.parseLong(config.get("maxSequences"));
		}
		
		System.out.println("Start Act Job");
		System.out.println("===============");
		System.out.println("* nn: "+Arrays.toString(nnNames));
		System.out.println("* pool: "+dataset);
		System.out.println("* environment: "+environment);
		System.out.println("* maxIterations: "+maxIterations);
		System.out.println("* maxSequences: "+maxSequences);
		System.out.println("---");
		
		
		Dictionary<String, Object> props = new Hashtable<>();
		String[] t = targets.stream().map(uuid -> uuid.toString()).collect(Collectors.toList()).toArray(new String[targets.size()]);
		props.put("targets", t);
		props.put("aiolos.unique", true);
		reg = coordinator.context.registerService(AgentListener.class, this, props);
		
		// start acting on each agent
		for(UUID target : targets){
			try {
				Agent agent = coordinator.agents.get(target);
				agents.put(target, agent);
				agent.act(environment, dataset, config, nnis.get(target));
			} catch(Throwable c){
				throw new JobFailedException(target, this.jobId, "Failed to start agent: "+c.getMessage(), c);
			}
		}
	}

	@Override
	public AgentResult getProgress() {
		return result;
	}
	
	@Override
	public void cleanup() {
		if(reg!=null)
			reg.unregister();
		
		for(Agent a : agents.values()){
			a.stop();
		}
	}

	@Override
	public void stop() throws Exception{
		if(started > 0){
			done(getProgress());
		} else {
			done(new JobFailedException(null, this.jobId, "Job "+this.jobId+" cancelled.", null));
		}
	}

	@Override
	public void onProgress(UUID agentId, AgentProgress progress) {
		if(deferred.getPromise().isDone()){
			return;
		}
		
		if(result.progress.get(agentId)==null){
			List<AgentProgress> p = new ArrayList<>();
			result.progress.put(agentId, p);
		}
		result.progress.get(agentId).add(progress);
		
		totalIterations += progress.iterations;
		totalSequences ++;
		
		int worker = 0;
		for(UUID id : result.progress.keySet()){
			if(!id.equals(agentId)){
				worker++;
			} else {
				break;
			}
		}
		coordinator.sendActProgress(this.jobId, worker, progress);
		
		if(maxIterations > 0 && totalIterations >= maxIterations){
			try {
				stop();
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
		
		if(maxSequences > 0 && totalSequences >= maxSequences){
			try {
				stop();
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
	}

	@Override
	public void onException(UUID agentId, Throwable e) {
		if(deferred.getPromise().isDone()){
			return;
		}
		done(new JobFailedException(agentId, this.jobId, "Agent failed: "+e.getMessage(), e));
	}

	@Override
	public void onFinish(UUID agentId) {
		if(deferred.getPromise().isDone()){
			return;
		}
		done(result);
	}
}
