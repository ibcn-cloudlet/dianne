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
package be.iminds.iot.dianne.command;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;

import org.apache.felix.service.command.Descriptor;
import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;

import be.iminds.iot.dianne.api.coordinator.DianneCoordinator;
import be.iminds.iot.dianne.api.coordinator.EvaluationResult;
import be.iminds.iot.dianne.api.coordinator.Job;
import be.iminds.iot.dianne.api.coordinator.LearnResult;
import be.iminds.iot.dianne.api.nn.eval.Evaluation;

/**
 * Separate component for learn commands ... should be moved to the command bundle later on
 */
@Component(
		service=Object.class,
		property={"osgi.command.scope=dianne",
				  "osgi.command.function=learn",
				  "osgi.command.function=eval",
				  "osgi.command.function=act",
				  "osgi.command.function=rl",
				  "osgi.command.function=running",
				  "osgi.command.function=queued",
				  "osgi.command.function=finished",
				  "osgi.command.function=job",
				  "osgi.command.function=jobs",
				  "osgi.command.function=stop"},
		immediate=true)
public class DianneCoordinatorCommands {

	private DianneCoordinator coordinator;
	
	@Descriptor("List running jobs.")
	public void running(){
		System.out.println("Running Jobs:");
		coordinator.runningJobs().stream().forEach(job -> printJob(job));
	}
	
	@Descriptor("List queued jobs.")
	public void queued(){
		System.out.println("Queued Jobs:");
		coordinator.queuedJobs().stream().forEach(job -> printJob(job));
	}
	
	@Descriptor("List (latest) finished jobs.")
	public void finished(){
		System.out.println("Finished Jobs:");
		coordinator.finishedJobs().stream().forEach(job -> printJob(job));
	}
	
	@Descriptor("List all jobs.")
	public void jobs(){
		queued();
		running();
		finished();
	}
	
	private void printJob(Job job){
		System.out.println(job.name+(job.name.equals(job.id.toString()) ? " " :" ("+job.id+") ")+"- Type: "+job.type+" - NNs: "+Arrays.toString(job.nn)+" - Dataset: "+job.dataset);
	}
	
	@Descriptor("Stop/cancel a job.")
	public void stop(
			@Descriptor("job uuid")
			String id){
		try {
			coordinator.stop(UUID.fromString(id));
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	@Descriptor("Stop all running jobs.")
	public void stop(){
		List<Job> running = coordinator.runningJobs();
		for(Job job : running){
			try {
				coordinator.stop(job.id);
			} catch(Exception e){}
		}
	}
	
	@Descriptor("Print job information.")
	public void job(
			@Descriptor("job uuid")
			String id){
		Job job = coordinator.getJob(UUID.fromString(id));
		if(job == null){
			System.out.println("Invalid job uuid");
		}
		
		System.out.println(job.name+" ("+job.id+")");
		System.out.println("---");
		System.out.println("Type: "+job.type);
		System.out.println("NNs: "+Arrays.toString(job.nn));
		System.out.println("Dataset: "+job.dataset);
		System.out.println("Runtimes: "+Arrays.toString(job.targets.toArray()));
		System.out.println("Configuration:");
		job.config.entrySet().stream().forEach(e -> System.out.println(e.getKey()+"="+e.getValue()));
	}
	
	@Descriptor("Train a neural network on a dataset. This uses the standard feedforward strategy by default.")
	public void learn(
			@Descriptor("Neural network name (use comma-separated list if multiple instances are required, e.g. for RL)")
			String nnName, 
			@Descriptor("Dataset to train on")
			String dataset, 
			@Descriptor("Additional properties, specified as key1=value1 key2=value2 ...")
			String... properties){
		try {
			Map<String, String> defaults = new HashMap<>();
			defaults.put("strategy", "FeedForwardLearningStrategy");
			
			Map<String, String> config = createConfig(defaults, properties);
		
			coordinator.learn(dataset, config, nnName.split(",")).then(p -> {
				System.out.println("Learn Job done!");
				LearnResult result = p.getValue();
				System.out.println("Iterations: "+result.getIterations());
				System.out.println("Last minibatch Error: "+result.getError());
				return null;
			}, p -> {
				System.out.println("Learn Job failed: "+p.getFailure().getMessage());
				p.getFailure().printStackTrace();
			});
		} catch(Exception e){
			e.printStackTrace();
		}
	}
	
	@Descriptor("Evaluate a neural network on a dataset. This uses a classification strategy by default.")
	public void eval(
			@Descriptor("Neural network name (use comma-separated list if multiple instances are required, e.g. for RL)")
			String nnName, 
			@Descriptor("Dataset to train on")
			String dataset, 
			@Descriptor("Additional properties, specified as key1=value1 key2=value2 ...")
			String... properties){
		try {
			Map<String, String> defaults = new HashMap<>();
			defaults.put("strategy", "ClassificationEvaluationStrategy");
			
			Map<String, String> config = createConfig(defaults, properties);
		
			coordinator.eval(dataset, config, nnName.split(",")).then(p -> {
				System.out.println("Evaluation Job done!");
				EvaluationResult result = p.getValue();
				for(Evaluation eval : result.evaluations.values()){
					System.out.println("Error: "+eval.error());
					System.out.println("Forward time: "+eval.forwardTime()+" ms");
				}
				return null;
			}, p -> {
				System.out.println("Evaluation Job failed: "+p.getFailure().getMessage());
				p.getFailure().printStackTrace();
			});
		} catch(Exception e){
			e.printStackTrace();
		}
	}
	
	@Descriptor("Start acting on an environment using a NN as agent. Push the generated samples to an experience pool. \n"
			+ "Uses an epsilon-greedy action selection strategy by default.")
	public void act(
			@Descriptor("Neural network name (use comma-separated list if multiple instances are required, e.g. for RL)")
			String nnName, 
			@Descriptor("Environment to act on")
			String environment, 
			@Descriptor("Experiencepool to push samples to")
			String experiencePool, 
			@Descriptor("Additional properties, specified as key1=value1 key2=value2 ...")
			String... properties){
		try {
			Map<String, String> defaults = new HashMap<>();
			defaults.put("strategy", "GreedyActionStrategy");
			defaults.put("environment", environment);
			
			Map<String, String> config = createConfig(defaults, properties);
		
			coordinator.act(experiencePool, config, nnName==null ? null : nnName.split(",")).then(p -> {
				System.out.println("Act Job done!");
				return null;
			}, p -> {
				System.out.println("Act Job failed: "+p.getFailure().getMessage());
				p.getFailure().printStackTrace();
			});
		} catch(Exception e){
			e.printStackTrace();
		}
	}
	
	@Descriptor("Start acting on an environment using a NN as agent, pushing samples to an experience pool. At the same time, \n"
			+ "start training the NN using the generated samples in the experience pool. The learner uses a DQN strategy by default.")
	public void rl(
			@Descriptor("Neural network name (use comma-separated list if multiple instances are required, e.g. for RL)")
			String nnName, 
			@Descriptor("Environment to act on")
			String environment, 
			@Descriptor("Experiencepool to use")
			String experiencePool,
			@Descriptor("Additional properties, specified as key1=value1 key2=value2 ...")
			String... properties){
		act(nnName, environment, experiencePool, properties);
		// wait a bit so the agent can configure the xp pool if required
		try {
			Thread.sleep(2000);
		} catch(Exception e){}
		rlearn(nnName, experiencePool, properties);
	}
	
	// placeholder for default DQN learn job
	private void rlearn(String nnName, String experiencePool, String... properties){
		try {
			Map<String, String> defaults = new HashMap<>();
			defaults.put("strategy", "be.iminds.iot.dianne.rl.learn.strategy.DeepQLearningStrategy");
			
			Map<String, String> config = createConfig(defaults, properties);
		
			coordinator.learn(experiencePool, config, nnName.split(",")).then(p -> {
				System.out.println("RL Learn Job done!");
				LearnResult result = p.getValue();
				System.out.println("Iterations: "+result.getIterations());
				System.out.println("Last minibatch Error: "+result.getError());
				return null;
			}, p -> {
				System.out.println("Learn Job failed: "+p.getFailure().getMessage());
			});
		} catch(Exception e){
			e.printStackTrace();
		}
	}
	
	private Map<String, String> createConfig(Map<String, String> defaults, String[] properties){
		Map<String, String> config = new HashMap<String, String>(defaults);
		for(String property : properties){
			String[] p = property.split("=");
			if(p.length==2){
				config.put(p[0].trim(), p[1].trim());
			}
		}
		
		return config;
	}
	
	@Reference
	void setDianneCoordinator(DianneCoordinator c){
		this.coordinator = c;
	}
	
}
