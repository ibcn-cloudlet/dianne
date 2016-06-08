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
package be.iminds.iot.dianne.rl.command;

import java.io.File;
import java.io.PrintStream;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;
import org.osgi.service.component.annotations.ReferenceCardinality;
import org.osgi.service.component.annotations.ReferencePolicy;

import be.iminds.iot.dianne.api.nn.learn.Learner;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkInstanceDTO;
import be.iminds.iot.dianne.api.nn.platform.DiannePlatform;
import be.iminds.iot.dianne.api.rl.agent.Agent;
import be.iminds.iot.dianne.api.rl.dataset.ExperiencePool;

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

	private DiannePlatform platform;
	
	private Agent agent;
	private Learner learner;
	private Map<String, ExperiencePool> pools = new HashMap<String, ExperiencePool>();

	
	public void rl(String nnName, String environment, String experiencePool, String... properties) throws Exception {
		Map<String, String> agentConfig = DianneRLAgentCommands.createAgentConfig(properties);
		Map<String, String> learnConfig = DianneRLLearnCommands.createLearnConfig(properties);
		
		NeuralNetworkInstanceDTO agentnni = platform.deployNeuralNetwork(nnName);
		NeuralNetworkInstanceDTO nni = platform.deployNeuralNetwork(nnName);
		NeuralNetworkInstanceDTO targeti = platform.deployNeuralNetwork(nnName);
		
		agent.act(experiencePool, agentConfig, agentnni, environment);
		
		learner.learn(experiencePool, learnConfig, nni, targeti);
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
			writer.println(i+"\t"+Arrays.toString(p.getState(i).get())+"\t"+Arrays.toString(p.getAction(i).get())+"\t"+p.getReward(i)+"\t"+(p.getNextState(i)==null ? "null" : Arrays.toString(p.getNextState(i).get())));
		}
	}
	
	@Reference
	void setAgent(Agent agent){
		this.agent = agent;
	}
	
	@Reference(target="(dianne.learner.category=RL)")
	void setLearner(Learner l){
		this.learner = l;
	}
	
	@Reference(cardinality = ReferenceCardinality.MULTIPLE, policy = ReferencePolicy.DYNAMIC)
	void addExperiencePool(ExperiencePool pool, Map<String, Object> properties) {
		String name = (String) properties.get("name");
		this.pools.put(name, pool);
	}

	void removeExperiencePool(ExperiencePool pool, Map<String, Object> properties) {
		String name = (String) properties.get("name");
		this.pools.remove(name);
	}
	
	@Reference
	void setDiannePlatform(DiannePlatform p){
		this.platform = p;
	}
}
