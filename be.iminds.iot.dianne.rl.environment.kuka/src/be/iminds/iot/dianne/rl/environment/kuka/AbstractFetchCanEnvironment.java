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

import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.TimeoutException;

import org.osgi.util.promise.Promise;

import be.iminds.iot.dianne.nn.util.DianneConfigHandler;
import be.iminds.iot.dianne.rl.environment.kuka.config.FetchCanConfig;
import be.iminds.iot.robot.api.arm.Arm;
import be.iminds.iot.simulator.api.Orientation;
import be.iminds.iot.simulator.api.Position;


/**
 * This environment sets up the scene to fetch a can
 * @author tverbele
 *
 */
public abstract class AbstractFetchCanEnvironment extends AbstractKukaEnvironment {
	
	protected static final float MAX_DISTANCE = 2.4f;
	protected static final float GRIP_DISTANCE = 0.565f;
	protected static final float EPSILON = 1e-4f;
	
	protected FetchCanConfig config;

	protected Random r = new Random(System.currentTimeMillis());

	protected float previousDistance;
	
	protected boolean grip = false;


	@Override
	protected float calculateReward() throws Exception {
		// calculate reward based on simulator info
		if(simulator != null){
			// if collision or can is too close
			if(checkCollisions()) {
				if (config.collisionTerminal) {
					terminal = true;
				} else {
					return -1.0f;
				}
			}
			
			float distance, canHeight;
			if (config.grip) {
				Position p = simulator.getPosition("can_ref", "youBot_ref");
				canHeight = p.z;
				// calculate distance of gripper tip relative to can
				p = simulator.getPosition("can_ref", "youBot_positionTip");
				distance = (float)Math.sqrt(p.x*p.x+p.y*p.y+p.z*p.z);
			} else {
				// calculate distance of youBot relative to can
				Position p = simulator.getPosition("can_ref", "youBot_ref");
				distance = (float)Math.hypot(p.x, p.y);
				canHeight = p.z;
			}

			// can is invisible to laser and can't be gripped anymore
			if (canHeight < -0.003) {
				terminal = true;
			}
			
			// max reward in radius of can by setting the distance to 0
			if(distance <= config.margin) {
				distance = 0.0f;
			}
			
			// if grip give reward according to position relative to can
			if(grip){
				if(config.earlyStop){
					// use position only
					if(distance <= config.margin){
						// succesful grip, mark as terminal
						terminal = true;
						return 1.0f * config.gripRewardScale;
					} 
				} else {
					// simulate actual grip action
					if(canHeight > 0){
						// can is lifted, reward 1 and mark as terminal
						terminal = true;
						return 1.0f * config.gripRewardScale;
					} 
				}
				
				grip = false;
				
				// punish wrong gripping?
				if(config.punishWrongGrip)
					return -1.0f * config.gripRewardScale;
			}
			
			float r;
			// also give intermediate reward for each action?
			if(config.intermediateReward) {
				if(config.relativeReward){
					// give +1 if closer -1 if further
					r = previousDistance - distance;
					if(config.discreteReward){
						r = r > EPSILON ? 1 : r < -EPSILON ? -1 : 0;
					} else {
						// boost it a bit
						r *= config.relativeRewardScale/(config.skip+1);
					}
				} else {
					// linear or exponential decaying reward function
					if (config.exponentialDecayingReward)
						// wolfram function: plot expm1(-a*x) + b with a=2.5 and b=1 for x = 0..2.4
						// where x: previousDistance, a: absoluteRewardScale, b: maxReward and expm1 =  e^x -1
						r = ((float)Math.expm1( -config.exponentialDecayingRewardScale * previousDistance));
					else {
						r = - previousDistance / MAX_DISTANCE;
					}
				}
				
				if (config.grip) {
					if (canHeight > 0) { // can is lifted if height is higher then 0
						r += canHeight;
						r *= config.gripRewardScale;
					}
					if (canHeight > 0.10) {
						r = 100;
						terminal = true;
					}
				}
				
				// reward offset
				r += config.maxReward;
			} else {
				r=0.0f;
			}
			previousDistance = distance;
			return r;
		} 
		
		// in case no simulator ... return reward variable that might be set manually
		return reward;
	}
	
	protected void initSimulator() throws Exception {
		boolean init = false;
		do {
			resetEnvironment();
			
			simulator.start(config.tick);
			if(config.tick){
				simulator.tick();
			}
			
			init = !checkCollisions();
			
			if(!init){
				simulator.stop();
				Thread.sleep(200);
			}
			
		} while(!init);
		
		
		// TODO there might be an issue with range sensor not coming online at all
		// should be fixed in robot project?
		long start = System.currentTimeMillis();
		while(kukaArm == null 
				|| kukaPlatform == null
				|| (!super.config.simState && rangeSensors.size() !=  1 + config.environmentSensors)){
			try {
				Thread.sleep(100);
				if(config.tick){
					simulator.tick();
				}
			} catch (InterruptedException|TimeoutException e) {}
			
			if(System.currentTimeMillis()-start > config.timeout){
				System.out.println("Failed to initialize youbot/laserscanner in environment... Try again");
				throw new Exception("Failed to initialize Kuka environment");
			}
		}
		
		try {
			Thread.sleep(1000);
		} catch (InterruptedException e) {} 
		
		// reset arm to candle
		Promise<Arm> p = kukaArm.setPositions(2.92510465f, 1.103709733f, -2.478948503f, 1.72566195f, 2.765485f);
		// simulate an iteration further
		while(!p.isDone() && System.currentTimeMillis()-start <= config.timeout) {
			if (config.tick) {
				simulator.tick();
			} else {
				Thread.sleep(100);
			}
		}
		
		// calculate reward here to initialize previousDistance
		calculateReward();
	}
	
	protected void resetEnvironment(){
		// in simulation we can control the position of the youbot and can
		// random init position and orientation of the robot
		resetYoubot();
					
		// set random can position
		resetCan();
	}
	
	protected void resetYoubot(){
		float x,y,o;
		
		switch (config.difficulty) {
		case FetchCanConfig.FIXED:
		case FetchCanConfig.WORKSPACE:
		case FetchCanConfig.VISIBLE:
			x = 0;
			y = 0;
			o = 0;
			break;
		case FetchCanConfig.RANDOM:
		default:
			x = (r.nextFloat()-0.5f);
			y = (r.nextFloat()-0.5f)*1.8f;
			o = (r.nextFloat()-0.5f)*6.28f;
			break;
		}
		simulator.setPosition("youBot", new Position(x, y, 0.0957f));
		simulator.setOrientation("youBot", new Orientation(-1.5707963f, o, -1.5707965f));
	}
	
	protected void resetCan(){
		float x,y;
		
		switch (config.difficulty) {
		case FetchCanConfig.FIXED:
			x = 0;
			y = GRIP_DISTANCE;
			break;
		case FetchCanConfig.WORKSPACE:
			// start position in front in workspace
			float d = r.nextFloat()*0.25f;
			double a = (r.nextFloat()-0.5f)*Math.PI;
			x = (float)Math.sin(a)*d;
			y = 0.4f + (float)Math.cos(a)*d;
			break;
		case FetchCanConfig.VISIBLE:
			x = (r.nextFloat()-0.5f)*1.6f;
			y = (0.125f + 3*r.nextFloat()/8f)*2.4f;
			break;
		case FetchCanConfig.RANDOM:
		default:
			x = (r.nextFloat()-0.5f)*1.6f;
			y = (r.nextFloat()-0.5f)*2.4f;
		}

		simulator.setOrientation("Can1", new Orientation(0, 0 ,1.6230719f));
		simulator.setPosition("Can1", new Position(x, y, 0.06f));
		
	}
	
	
	@Override
	public void configure(Map<String, String> config) {
		this.config = DianneConfigHandler.getConfig(config, FetchCanConfig.class);

		if(this.config.seed != 0){
			r = new Random(this.config.seed);
		}
		
		// configure the simulated environment
		if(simulator != null){
			Map<String, String> entities = new HashMap<String, String>();
			entities.put("youBot", "be.iminds.iot.robot.youbot.ros.Youbot");
			if(!super.config.simState)
				entities.put("hokuyo", "be.iminds.iot.sensor.range.ros.LaserScanner");

			// only activate configured sensors
			simulator.setProperty("hokuyo", "active", false);
			simulator.setProperty("hokuyo#0", "active", false);
			simulator.setProperty("hokuyo#1", "active", false);
			simulator.setProperty("hokuyo#2", "active", false);
			simulator.setProperty("hokuyo#3", "active", false);
			
			switch(this.config.environmentSensors){
			case 0:
				break;
			case 1:
				entities.put("hokuyo#0", "be.iminds.iot.sensor.range.ros.LaserScanner");
				break;
			case 2:
				entities.put("hokuyo#0", "be.iminds.iot.sensor.range.ros.LaserScanner");
				entities.put("hokuyo#3", "be.iminds.iot.sensor.range.ros.LaserScanner");
				break;
			case 4:
				entities.put("hokuyo#0", "be.iminds.iot.sensor.range.ros.LaserScanner");
				entities.put("hokuyo#1", "be.iminds.iot.sensor.range.ros.LaserScanner");
				entities.put("hokuyo#2", "be.iminds.iot.sensor.range.ros.LaserScanner");
				entities.put("hokuyo#3", "be.iminds.iot.sensor.range.ros.LaserScanner");
				break;
			default:
				System.out.println("Invalid number of environment sensors given: "+this.config.environmentSensors+", should be 0,1,2 or 4");
			}
			
			simulator.loadScene("scenes/youbot_fetch_can.ttt", entities);
			
			for(String key : entities.keySet()){
				if(key.startsWith("hokuyo")){
					simulator.setProperty(key, "active", true);
					simulator.setProperty(key, "scanPoints", super.config.scanPoints);
					simulator.setProperty(key, "showLaser", super.config.showLaser);
				}
			}
		} 
		
	}
	

}
