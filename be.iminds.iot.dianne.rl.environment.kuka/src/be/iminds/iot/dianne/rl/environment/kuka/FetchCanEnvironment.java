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

import java.util.Map;

import org.osgi.service.component.annotations.Component;
import org.osgi.util.promise.Promise;

import be.iminds.iot.dianne.api.rl.environment.Environment;
import be.iminds.iot.dianne.nn.util.DianneConfigHandler;
import be.iminds.iot.dianne.rl.environment.kuka.api.KukaEnvironment;
import be.iminds.iot.dianne.rl.environment.kuka.config.FetchCanConfig;
import be.iminds.iot.dianne.rl.environment.kuka.config.FetchCanConfig.Difficulty;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;
import be.iminds.iot.robot.api.arm.Arm;
import be.iminds.iot.simulator.api.Orientation;
import be.iminds.iot.simulator.api.Position;


@Component(immediate = true,
	service = {Environment.class, KukaEnvironment.class},
	property = { "name="+FetchCanEnvironment.NAME, 
				 "aiolos.unique=true",
				 "aiolos.combine=*",
				 "osgi.command.scope=fetchcan",
				 "osgi.command.function=start",
				 "osgi.command.function=stop",
				 "osgi.command.function=pause",
				 "osgi.command.function=resume",
				 "osgi.command.function=reward",
				 "osgi.command.function=resetCan"})
public class FetchCanEnvironment extends AbstractKukaEnvironment {
	
	public static final String NAME = "FetchCan";
	
	protected static final float MAX_DISTANCE = 3f;
	protected static final float GRIP_DISTANCE = 0.565f;
	protected static final float EPSILON = 1e-4f;
	
	protected FetchCanConfig config;

	protected float previousDistance;
	protected boolean grip = false;
	
	// default can ref height - will be updated with the actual ref height in resetCan().
	private float canRefInitHeight = 0.5f;
	
	@Override
	public void configure(Map<String, String> config) {
		this.config = DianneConfigHandler.getConfig(config, FetchCanConfig.class);
	}
	
	@Override
	public String getName(){
		return NAME;
	}

	@Override
	public int[] actionDims() {
		return new int[]{7};
	}
	
	@Override
	protected void executeAction(Tensor a) throws Exception {
		int action = TensorOps.argmax(a);
		
		switch(action){
		case 0:
			kukaPlatform.move(0f, config.speed, 0f);
			break;
		case 1:
			kukaPlatform.move(0f, -config.speed, 0f);
			break;
		case 2:
			kukaPlatform.move(config.speed, 0f, 0f);
			break;
		case 3:
			kukaPlatform.move(-config.speed, 0f, 0f);
			break;
		case 4:
			kukaPlatform.move(0f, 0.f, 2*config.speed);
			break;
		case 5:
			kukaPlatform.move(0f, 0.f, -2*config.speed);
			break;	
		case 6:
			grip = true;
			
			kukaPlatform.stop();	

			if(config.earlyStop){
				break;
			}
			
			Promise<Arm> result = kukaArm.openGripper()
				.then(p -> kukaArm.setPositions(2.92f, 0.0f, 0.0f, 0.0f, 2.875f))
				.then(p -> kukaArm.setPositions(2.92f, 1.76f, -1.37f, 2.55f))
				.then(p -> kukaArm.closeGripper())
				.then(p -> kukaArm.setPositions(0.01f, 0.8f))
				.then(p -> kukaArm.setPositions(0.01f, 0.8f, -1f, 2.9f))
				.then(p -> kukaArm.openGripper())
				.then(p -> kukaArm.setPosition(1, -1.3f))
				.then(p -> kukaArm.reset());
			
			
			// in simulation keep on ticking to let the action complete
			if(simulator != null){
				for(int i=0;i<130;i++){
					simulator.tick();
					
					// stop when colliding
					if(simulator.checkCollisions("Border")){
						return;
					}
				}
			} else {
				// wait until grip is done
				result.getValue();
			}
			
			break;
		}
		
		// simulate an iteration further
		if(simulator != null){
			for(int i=0;i<=config.skip;i++){
				simulator.tick();
			}
		}
	}
	
	
	@Override
	protected float calculateReward() throws Exception {
		// calculate reward based on simulator info
		if(simulator != null){
			// if collision or can is too close
			if(checkCollisions()) {
				// end sequence with original reward OR return negative reward
				if (config.collisionTerminal) {
					terminal = true;
				} else {
					return -1.0f;
				}
			}
			
			float distance, canHeight;
			canHeight = simulator.getPosition("can_ref").z - this.canRefInitHeight;
			if (config.grip) {
				// calculate distance of gripper tip relative to can
				Position p = simulator.getPosition("can_ref", "youBot_positionTip");
				distance = (float)Math.sqrt(p.x*p.x+p.y*p.y+p.z*p.z);
			} else {
				// calculate distance of youBot relative to can
				Position p = simulator.getPosition("can_ref", "youBot_positionTargetCan");
				distance = (float)Math.hypot(p.x, p.y);
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
						// wolfram function: plot expm1(-a*|x|) + b with a=2.5 and b=1 for x = -2..2
						// sharp point at max value
						// where x: previousDistance, a: exponentialDecayingRewardScale, b: rewardOffset and expm1 =  e^x -1
						r = ((float)Math.expm1( -config.exponentialDecayingRewardScale * previousDistance));
					else {
						r = - previousDistance / MAX_DISTANCE;
						if(r < -1){
							r = -1;
						}
					}
				}
			} else {
				r=0.0f;
			}

			// reward offset and scale
			r += config.rewardOffset;
			r *= config.rewardScale;
			
			previousDistance = distance;
			return r;
		} 
		
		// in case no simulator ... return reward variable that might be set manually
		return reward;
	}
	
	protected void resetEnvironment(){
		int plane;
		if(config.difficulty==Difficulty.START_DOCKED
				|| config.difficulty==Difficulty.RANDOM_DOCK){
			plane = r.nextInt(2)+1;
		} else {
			plane = 0;
		}
		
		// set plane and/or youBot position
		Position p;
		switch(plane){
		case 0:
			simulator.setOrientation("Plane1", new Orientation(0, (float)Math.PI/2, (float)Math.PI/2));
			simulator.setPosition("Plane1", new Position(-1, -0.9f, 0.15f));
			simulator.setPosition("dock_ref", new Position(-0.6f, -1.5f, 0f));
			
			resetYoubot();
			break;
		case 1:
			simulator.setOrientation("Plane1", new Orientation(0, (float)Math.PI/2, (float)Math.PI/2));
			float x = -0.3f + r.nextFloat()/10;
			simulator.setPosition("Plane1", new Position(x, -0.9f, 0.15f));
			simulator.setPosition("dock_ref", new Position(-0.6f, -1.5f, 0f));
			
			if(config.difficulty == Difficulty.START_DOCKED){
				simulator.setPosition("youBot", new Position(-0.583f, -0.939f, 0.0957f));
				simulator.setOrientation("youBot", new Orientation(1.5707963f, 0, 1.5707965f));
			} else {
				do {
					resetYoubot();
					p = simulator.getPosition("youBot");
				} while(p.x < 0.1 && p.y < -0.1);
			}
			break;
		case 2:
			float y = -0.6f + (r.nextFloat()-0.5f)/5;
			simulator.setOrientation("Plane1", new Orientation((float)Math.PI/2, 0, 0));
			simulator.setPosition("Plane1", new Position(-0.45f, y, 0.15f));
			simulator.setPosition("dock_ref", new Position(-1f, -1f, 0f));
			
			if(config.difficulty == Difficulty.START_DOCKED){
				simulator.setPosition("youBot", new Position(-0.431f, -1f, 0.0957f));
				simulator.setOrientation("youBot", new Orientation(-1.5707963f, -1.5707963f, -1.5707965f));
			} else {
				do {
					resetYoubot();
					p = simulator.getPosition("youBot");
				} while(p.y < -0.2);
			}
			break;
		}
		
		// set random can position
		resetCan();
	}
	
	protected void resetYoubot(){
		float x,y,o;
		
		switch (config.difficulty) {
		case FIXED:
		case WORKSPACE:
		case VISIBLE:
			x = 0;
			y = 0;
			o = 0;
			break;
		case RANDOM:
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
		
		float s = 0;
		while(s < 0.15f) { // can should not be colliding with youbot from start	
		
			switch (config.difficulty) {
			case FIXED:
				x = 0;
				y = GRIP_DISTANCE;
				break;
			case WORKSPACE:
				// start position in front in workspace
				float d = r.nextFloat()*0.25f;
				double a = (r.nextFloat()-0.5f)*Math.PI;
				x = (float)Math.sin(a)*d;
				y = 0.4f + (float)Math.cos(a)*d;
				break;
			case VISIBLE:
				x = (r.nextFloat()-0.5f)*1.6f;
				y = (0.125f + 3*r.nextFloat()/8f)*2.4f;
				break;
			case RANDOM:
			default:
				x = (r.nextFloat()-0.5f)*1.6f;
				y = (r.nextFloat()-0.5f)*2.4f;
			}
	
			simulator.setOrientation("Can1", new Orientation(0, 0 ,1.6230719f));
			simulator.setPosition("Can1", new Position(x, y, 0.06f));
			
			Position d = simulator.getPosition("Can1", "youBot");
			s = d.y*d.y+d.z*d.z;
		}
		canRefInitHeight = simulator.getPosition("can_ref").z;
	}
}
