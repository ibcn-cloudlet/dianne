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
package be.iminds.iot.dianne.rl.environment.kuka;

import java.util.Collections;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import org.osgi.service.component.annotations.Reference;
import org.osgi.service.component.annotations.ReferenceCardinality;
import org.osgi.service.component.annotations.ReferencePolicy;

import be.iminds.iot.dianne.api.rl.environment.Environment;
import be.iminds.iot.dianne.api.rl.environment.EnvironmentListener;
import be.iminds.iot.dianne.rl.environment.kuka.api.KukaEnvironment;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.robot.api.Arm;
import be.iminds.iot.robot.api.OmniDirectional;
import be.iminds.iot.sensor.api.LaserScanner;
import be.iminds.iot.simulator.api.Simulator;

/**
 * Abstract Kuka environment that can be used as base for running the Kuka with
 * laser range scanner in our cage
 * 
 * In case no simulator is available, the environment will pause after each termination.
 * 
 * Make sure to configure the properties in your inheriting components:
 * 
@Component(immediate = true,
	service = {Environment.class, KukaEnvironment.class},
	property = { "name="+<your environment name>, 
				 "aiolos.unique=be.iminds.iot.dianne.api.rl.Environment",
				 "aiolos.combine=*",
				 "osgi.command.scope=kukaagent",
				 "osgi.command.function=rest",
				 "osgi.command.function=go",
				 "osgi.command.function=reward"})
 * 
 * @author tverbele
 *
 */
public abstract class AbstractKukaEnvironment implements Environment, KukaEnvironment {
	
	private Set<EnvironmentListener> listeners = Collections.synchronizedSet(new HashSet<>());
	
	protected volatile boolean active = false;
	protected boolean terminal = false;
	protected Tensor observation;
	
	// TODO for now limited to 1 youbot, 1 laserscanner
	protected OmniDirectional kukaPlatform;
	protected Arm kukaArm;
	protected LaserScanner rangeSensor;
	
	// Environment can be both simulated or on real robot
	protected Simulator simulator;
	
	protected float reward = 0;
	private volatile boolean pause = false;
	
	public abstract String getName();
	
	@Override
	public int[] observationDims() {
		// observation from laser range scanner
		return new int[]{512};
	}

	@Override
	public abstract int[] actionDims();
	
	@Override
	public float performAction(Tensor action) {
		if(!active)
			throw new RuntimeException("The Environment is not active!");
		
		// execute action
		try {
			executeAction(action);
		} catch (Exception e) {
			throw new RuntimeException("Failed executing action "+action);
		}
		
		if(simulator == null && terminal){
			// pause the agent on termination to enter reward
			pause = true;
			System.out.println("Enter your reward for this episode (type \"reward x\" in CLI with x your reward as floating point)");
		}
		
		// calculate reward
		try {
			calculateReward();
		} catch(Exception e){
			throw new RuntimeException("Failed calculating reward");
		}
		
		// pause in case one wants to set reward
		if(pause){
			kukaPlatform.stop();
			waitForResume();
		}
		
		updateObservation();

		synchronized(listeners){
			listeners.stream().forEach(l -> l.onAction(reward, observation));
		}
			
		return reward;
	}
	
	@Override
	public Tensor getObservation(Tensor t) {
		if(!active)
			throw new RuntimeException("The Environment is not active!");
		
		if(terminal){
			return null;
		}
		return observation.copyInto(t);
	}

	@Override
	public void reset() {
		if(!active)
			throw new RuntimeException("The Environment is not active!");
		
		terminal = false;
		
		// TODO handle failure here?
		try {
			deinit();

			init();
		} catch(Exception e){
			throw new RuntimeException("Failed to initialize the environment ...", e);
		}
		
		updateObservation();
		
		listeners.stream().forEach(l -> l.onAction(0, observation));
	}
	
	
	protected abstract void executeAction(Tensor a) throws Exception; 
	
	protected abstract float calculateReward() throws Exception;
	
	protected abstract void initSimulator() throws Exception;
	
	protected abstract void configure(Map<String, String> config);
		
	public void reward(float r){
		this.reward = r;
		go();
	}
	
	public void rest(){
		pause = true;
	}
	
	public void go(){
		pause = false;
		synchronized(this){
			this.notifyAll();
		}
	}
	
	private void waitForResume(){
		synchronized(this){
			if(pause){
				try {
					this.wait();
				} catch (InterruptedException e) {
					System.out.println("Environment interrupted!");
				}
				pause = false;
			}
		}
	}
	
	private void updateObservation(){
		float[] data = rangeSensor.getValue().data;
		observation = new Tensor(data, data.length);
	}
	
	
	private void init() throws Exception {
		if(simulator == null) {
			// automatically pause the environment until the user resumes from CLI
			pause = true;
			System.out.println("Reset your environment and resume by typing the \"go\" command.");
			waitForResume();
			
		} else {
			initSimulator();
		}
	}
	
	private void deinit(){
		if(simulator != null){
			simulator.stop();
		}
	}
	

	@Reference(cardinality = ReferenceCardinality.MULTIPLE, policy = ReferencePolicy.DYNAMIC)
	void addEnvironmentListener(EnvironmentListener l, Map<String, Object> properties){
		String target = (String) properties.get("target");
		if(target==null || target.equals(getName())){
			listeners.add(l);
		}
	}
	
	void removeEnvironmentListener(EnvironmentListener l){
		listeners.remove(l);
	}

	
	// TODO use target filters for these  (involves spawning environments from configadmin?)
	@Reference(cardinality=ReferenceCardinality.OPTIONAL, policy=ReferencePolicy.DYNAMIC)
	void setArm(Arm a){
		this.kukaArm = a;
	}
	
	void unsetArm(Arm a){
		this.kukaArm = null;
	}
	
	@Reference(cardinality=ReferenceCardinality.OPTIONAL, policy=ReferencePolicy.DYNAMIC)
	void setPlatform(OmniDirectional o){
		this.kukaPlatform = o;
	}
	
	void unsetPlatform(OmniDirectional o){
		if(this.kukaPlatform==o)
			this.kukaPlatform = null;
	}
	
	@Reference(cardinality=ReferenceCardinality.OPTIONAL, policy=ReferencePolicy.DYNAMIC)
	void setLaserScanner(LaserScanner l){
		this.rangeSensor = l;
	}
	
	void unsetLaserScanner(LaserScanner l){
		if(l == this.rangeSensor)
			this.rangeSensor = null;
	}
	
	@Reference(cardinality=ReferenceCardinality.OPTIONAL, policy=ReferencePolicy.DYNAMIC)
	void setSimulator(Simulator s){
		this.simulator = s;
	}
	
	void unsetSimulator(Simulator s){
		if(this.simulator == s){
			this.simulator = s;
		}
	}
	
	@Override
	public void setup(Map<String, String> config) {
		if(active)
			throw new RuntimeException("This Environment is already active");

		configure(config);
		
		active = true;
		
		reset();
	}
	
	@Override
	public void cleanup() {
		active = false;
		
		deinit();
	}
	
}