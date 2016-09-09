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
import java.util.Random;
import java.util.Set;

import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;
import org.osgi.service.component.annotations.ReferenceCardinality;
import org.osgi.service.component.annotations.ReferencePolicy;

import be.iminds.iot.dianne.api.rl.environment.Environment;
import be.iminds.iot.dianne.api.rl.environment.EnvironmentListener;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;
import be.iminds.iot.robot.api.Arm;
import be.iminds.iot.robot.api.OmniDirectional;
import be.iminds.iot.sensor.api.LaserScanner;
import be.iminds.iot.simulator.api.Orientation;
import be.iminds.iot.simulator.api.Position;
import be.iminds.iot.simulator.api.Simulator;


@Component(immediate = true,
	property = { "name="+FetchCanEnvironment.NAME, "aiolos.unique=be.iminds.iot.dianne.api.rl.Environment" })
public class FetchCanEnvironment implements Environment {
	
	public static final String NAME = "Kuka";
	
	private Set<EnvironmentListener> listeners = Collections.synchronizedSet(new HashSet<>());
	
	private volatile boolean active = false;
	private boolean terminal = false;
	private Tensor observation;
	
	private OmniDirectional kukaPlatform;
	private Arm kukaArm;
	private LaserScanner rangeSensor;
	
	private Simulator simulator;
	
	private Random r = new Random(System.currentTimeMillis());
	
	@Override
	public int[] observationDims() {
		return new int[]{512};
	}

	@Override
	public int[] actionDims() {
		return new int[]{7};
	}
	
	@Override
	public float performAction(Tensor action) {
		if(!active)
			throw new RuntimeException("The Environment is not active!");
		
		
		// execute action and calculate reward
		int a = TensorOps.argmax(action);

		executeAction(a);
		
		float reward = calculateReward();
		
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
	
	
	private void executeAction(int action){

		switch(action){
		case 0:
			kukaPlatform.move(0f, 0.2f, 0f);
			break;
		case 1:
			kukaPlatform.move(0f, -0.2f, 0f);
			break;
		case 2:
			kukaPlatform.move(0.2f, 0f, 0f);
			break;
		case 3:
			kukaPlatform.move(-0.2f, 0f, 0f);
			break;
		case 4:
			kukaPlatform.move(0f, 0.f, 0.4f);
			break;
		case 5:
			kukaPlatform.move(0f, 0.f, -0.4f);
			break;	
		case 6:
			terminal = true;
			
			kukaPlatform.stop();	

			// stop early when we are nowhere near the pick location (further than 10cm)
			Position d = simulator.getPosition("Can1", "youBot");
			if(Math.abs(d.y) > 0.1 || Math.abs(d.z - 0.58) > 0.1){
				return;
			}
			
			kukaArm.openGripper()
				.then(p -> kukaArm.setPosition(0, 2.92f))
				.then(p -> kukaArm.setPosition(4, 2.875f))
				.then(p -> kukaArm.setPositions(2.92f, 1.76f, -1.37f, 2.55f))
				.then(p -> kukaArm.closeGripper())
				.then(p -> kukaArm.setPositions(0.01f, 0.8f))
				.then(p -> kukaArm.setPositions(0.01f, 0.8f, -1f, 2.9f))
				.then(p -> kukaArm.openGripper())
				.then(p -> kukaArm.setPosition(1, -1.3f))
				.then(p -> kukaArm.reset());
			
			// keep on ticking to let the action complete
			for(int i=0;i<300;i++){
				simulator.tick();
				
				// stop when colliding
				if(simulator.checkCollisions("Border")){
					return;
				}
			}
			return;
		}
		
		// simulate an iteration further
		simulator.tick();	
	
	}
	
	private float previousDistance = 0.0f;
	
	private float calculateReward(){
		// in case of collision, reward -1
		// in case of succesful grip, reward 1, insuccesful grip, -1
		// else, reward between 0 and 0.5 as one gets closer to the optimal grip point
		if(simulator.checkCollisions("Border")){
			terminal = true;
			return -1.0f;
		}

		Position d = simulator.getPosition("Can1", "youBot");
		// if terminal, check for grip success
		if(terminal){
			if(d.x > 0){
				// can is lifted, reward 1
				return  1.0f;
			} else {
				// else failed, no reward
				return 0.0f;
			}
		}

		float dx = d.y;
		float dy = d.z - 0.58f;

		// dy should come close to 0.58 for succesful grip
		// dx should come close to 0
		float d2 = dx*dx + dy*dy;
		float distance = (float)Math.sqrt(d2);
		
		// give reward based on whether one gets closer/further from target
		// rescale to get values (approx) between -0.25..0.25 
		float reward = (previousDistance-distance)*10;
		
		previousDistance = distance;
		return reward;
	}
	
	private void updateObservation(){
		float[] data = rangeSensor.getValue().data;
		observation = new Tensor(data, data.length);
	}
	
	private void init() throws Exception {
		// TODO also random init for youbot position and orientation?
		
		// always start the youbot in 0,0 for now
		Position p = simulator.getPosition("youBot");
		simulator.setPosition("youBot", new Position(0, 0, p.z));
		
		// set random can position, right now in front of the youbot
		p = simulator.getPosition("Can1");
		float x = (r.nextFloat()-0.5f)*0.55f;
		x = x > 0 ? x + 0.25f : x - 0.25f; 
		float y = r.nextFloat()*0.9f+0.4f;
		simulator.setPosition("Can1", new Position(x, y, 0.06f));
		simulator.setOrientation("Can1", new Orientation(0, 0 ,1.6230719f));
		
		simulator.start(true);
		
		// TODO there might be an issue with range sensor not coming online at all
		// should be fixed in robot project..
		simulator.tick();
		long start = System.currentTimeMillis();
		while(rangeSensor==null
				|| kukaArm == null 
				|| kukaPlatform == null){
			try {
				Thread.sleep(100);
				simulator.tick();
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			
			if(System.currentTimeMillis()-start > 30000){
				System.out.println("Failed to initialize youbot/laserscanner in environment... Try again");
				System.out.println("BLOCKED?!");
				//throw new Exception("Failed to initialize youbot/laserscanner in environment");
			}
		}
		
		// calculate reward here to initialize previousDistance
		calculateReward();
	}
	
	private void deinit(){
		simulator.stop();
	}
	

	@Reference(cardinality = ReferenceCardinality.MULTIPLE, policy = ReferencePolicy.DYNAMIC)
	void addEnvironmentListener(EnvironmentListener l, Map<String, Object> properties){
		String target = (String) properties.get("target");
		if(target==null || target.equals(NAME)){
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
	
	@Reference
	void setSimulator(Simulator s){
		this.simulator = s;
	}
	
	@Override
	public void setup(Map<String, String> config) {
		if(active)
			throw new RuntimeException("This Environment is already active");

		active = true;
		
		// configure the environment
		simulator.loadScene("youbot_fetch_can.ttt");
		
		reset();
	}
	
	@Override
	public void cleanup() {
		active = false;
		
		deinit();
	}
	
}
