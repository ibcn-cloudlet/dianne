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
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;
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
public class DockingEnvironment extends AbstractFetchCanEnvironment {
	
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
		if(simulator != null && super.config.tick){
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
			
			// calculate distance of youBot relative to dock point
			Position p = simulator.getPosition("dock_ref", "youBot_ref");
			float distance = (float)Math.hypot(p.x, p.y);
			
			// max reward in radius of can by setting the distance to 0
			if(distance <= config.margin) {
				distance = 0.0f;
			}
			
			// if stop give reward according to position relative to docking point
			// also give intermediate reward for each action?
			float r;
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

	@Override
	protected void resetEnvironment(){
		simulator.setPosition("Plane1", new Position(0.2925f, 0.9f, 0.15f));
		simulator.setPosition("dock_ref", new Position(0.6f, 0.9f, 0f));
		super.resetYoubot();
	}
}
