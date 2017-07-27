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

import org.osgi.service.component.annotations.Component;

import be.iminds.iot.dianne.api.rl.environment.Environment;
import be.iminds.iot.dianne.rl.environment.kuka.api.KukaEnvironment;
import be.iminds.iot.dianne.rl.environment.kuka.config.FetchCanConfig.Difficulty;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;
import be.iminds.iot.simulator.api.Orientation;
import be.iminds.iot.simulator.api.Position;


@Component(immediate = true,
	service = {Environment.class, KukaEnvironment.class},
	property = { "name="+DockingEnvironment.NAME, 
				 "aiolos.unique=true",
				 "aiolos.combine=*",
				 "osgi.command.scope=docking",
				 "osgi.command.function=start",
				 "osgi.command.function=stop",
				 "osgi.command.function=pause",
				 "osgi.command.function=resume",
				 "osgi.command.function=reward"})
public class DockingEnvironment extends FetchCanEnvironment {
	
	public static final String NAME = "Docking";
	
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
			kukaPlatform.stop();	
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
			if(simulator.checkCollisions("Border")) {
				// end sequence with original reward OR return negative reward
				if (config.collisionTerminal) {
					terminal = true;
				} else {
					return -1.0f;
				}
			}
			
			// calculate distance of youBot relative to dock point ... 
			// reuse positionTargetCan to encourage the robot to turn towards the dock point
			Position p = simulator.getPosition("dock_ref", "youBot_positionTargetCan");
			float distance = (float)Math.hypot(p.x, p.y);
			
			// max reward in radius of can by setting the distance to 0
			if(distance <= config.margin) {
				distance = 0.0f;
			}
			
			float r = 0;			
			switch(config.intermediateReward) {
			case NONE:
				r=0;
				break;
			case RELATIVE_DISCRETE:
				// give +1 if closer -1 if further
				float d = previousDistance - distance;
				r = d > EPSILON ? 1 : d < -EPSILON ? -1 : 0;
				break;
			case RELATIVE_CONTINUOUS:
				r = (previousDistance - distance)/(config.skip+1);
				break;
			case LINEAR:
				r = Math.max(- previousDistance / MAX_DISTANCE, -1);
				break;
			case EXPONENTIAL:
				// wolfram function: plot expm1(-a*|x|) + b with a=2.5 and b=1 for x = -2..2
				// sharp point around 0 distance
				// where x: previousDistance, a: exponentialDecayingRewardScale, b: rewardOffset and expm1 =  e^x -1
				r = ((float)Math.expm1( -config.distanceScale * previousDistance));
				break;
			case HYPERBOLIC:
				// wolfram function: plot -tanh(a*x)^2 + b with a=1.3 and b=1 for x = -2..2
				// smooth function around 0 distance
				// where x: previousDistance, a: exponentialDecayingRewardScale, b: rewardOffset
				r = (float) -Math.pow(Math.atan(config.distanceScale * previousDistance),2);
				break;
			default:
				// this should never happen and is a programming error.
				throw new UnsupportedOperationException(String.format("The reward function '%s' is currently unsupported", config.intermediateReward));
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
		super.config.difficulty = Difficulty.RANDOM_DOCK;
		super.resetEnvironment();
	}
	
	protected void resetCan(){
		// no can when docking?
		simulator.setOrientation("Can1", new Orientation(0, 0 ,1.6230719f));
		simulator.setPosition("Can1", new Position(-1.0f, 0.0f, 0.06f));
	}
}
