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

import be.iminds.iot.simulator.api.Orientation;
import be.iminds.iot.simulator.api.Position;


/**
 * This environment sets up the scene to fetch a can
 * @author tverbele
 *
 */
public abstract class AbstractGrabCanEnvironment extends AbstractFetchCanEnvironment {

	private int count = 0;
	private float previousDistance;

	@Override
	protected float calculateReward() throws Exception {
		if(count++ == config.maxActions){
			terminal = true;
			count = 0;
		}
		
		// calculate reward based on simulator info
		if(simulator != null){
			
			// if collision or can is too close
			if(simulator.checkCollisions("Border")){
				if (config.collisionTerminal) {
					terminal = true;
					count = 0;
				} else {
					return -1.0f;
				}
			}
			float r = 0.0f;
			Position d = simulator.getPosition("Can1", "Rectangle7");
			
			// calculate distance of gripper relative to can
			float distance = (float)Math.sqrt(d.x*d.x+d.y*d.y+d.z*d.z);
			r = - previousDistance;
			r += config.maxReward; // reward offset
			previousDistance = distance;
			return r;
		} 
		
		// in case no simulator ... return reward variable that might be set manually
		return reward;
	}
	
	public void resetYoubot(){
		float x = 0;
		float y = 0;
		float o = 0;
		simulator.setPosition("youBot", new Position(x, y, 0.0957f));
		simulator.setOrientation("youBot", new Orientation(-1.5707963f, o, -1.5707965f));
	}
	
	public void resetCan(){
		float x,y;
		
		// set random can position
		float s = 0;
		while(s < 0.15f) { // can should not be colliding with youbot from start	
			if(config.difficulty <= 0){
				x = 0;
			} else {
				x = (r.nextFloat()-0.5f)*4*GRIP_DISTANCE;
			}
			
			if(config.difficulty <= 0) {
				y = GRIP_DISTANCE;
			} else
			if(config.difficulty <= 1){
				y = (0.125f + 3*r.nextFloat()/8f)*2.4f;
			} else {
				y = (r.nextFloat()-0.5f)*2.4f;
			}

			simulator.setOrientation("Can1", new Orientation(0, 0 ,1.6230719f));
			simulator.setPosition("Can1", new Position(x, y, 0.06f));
			
			Position d = simulator.getPosition("Can1", "youBot");
			s = d.y*d.y+d.z*d.z;
		} 
	}
}
