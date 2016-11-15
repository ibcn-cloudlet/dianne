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

import be.iminds.iot.dianne.nn.util.DianneConfigHandler;
import be.iminds.iot.dianne.rl.environment.kuka.config.FetchCanConfig;
import be.iminds.iot.simulator.api.Orientation;
import be.iminds.iot.simulator.api.Position;


/**
 * This environment sets up the scene to fetch a can
 * @author tverbele
 *
 */
public abstract class AbstractFetchCanEnvironment extends AbstractKukaEnvironment {
	
	public static final String NAME = "Kuka";
	
	protected static final float MAX_DISTANCE = 2.4f;
	protected static final float GRIP_DISTANCE = 0.565f;
	
	protected FetchCanConfig config;
	
	protected Random r = new Random(System.currentTimeMillis());

	private float previousDistance;
	private int count = 0;
	
	protected boolean grip = false;


	@Override
	protected float calculateReward() throws Exception {
		if(count++ == config.maxActions){
			count = 0;
			terminal = true;
		}
		
		// calculate reward based on simulator info
		if(simulator != null){
			
			// in case of collision, reward -1
			// in case of succesful grip, reward 1, insuccesful grip, -1
			// else, reward between 0 and 0.5 as one gets closer to the optimal grip point
			if(simulator.checkCollisions("Border")){
				return -1.0f;
			}
	
			Position d = simulator.getPosition("Can1", "youBot");
			
			// calculate distance of youBot relative to can
			float dx = d.y;
			float dy = d.z - GRIP_DISTANCE;
	
			// if can is too close, give same reward as collision
			if(Math.abs(d.y) < 0.23 && Math.abs(d.z) < 0.37){
				return -1.0f;
			}
			
			// dy should come close to 0.565 for succesful grip
			// dx should come close to 0
			float d2 = dx*dx + dy*dy;
			float distance = (float)Math.sqrt(d2);
			
			
			// if grip give reward according to position relative to can
			if(grip){
				if(config.earlyStop){
					// use position only
					if(Math.abs(dx) <= config.margin
						&& Math.abs(dy) <= config.margin){
						// succesful grip, mark as terminal
						terminal = true;
						return 1.0f;
					} 
				} else {
					// simulate actual grip action
					if(d.x > 0){
						// can is lifted, reward 1 and mark as terminal
						terminal = true;
						return 1.0f;
					} 
				}
				
				grip = false;
			}
			
			// also give intermediate reward for each action?
			if(config.intermediateReward){
				float r = - previousDistance / MAX_DISTANCE;
				previousDistance = distance;
				return r;
			} else {
				return 0.0f;
			}
		} 
		
		// in case no simulator ... return reward variable that might be set manually
		return reward;
	}
	
	protected void initSimulator() throws Exception {
		// in simulation we can control the position of the youbot and can
		float x,y,o;
		
		// random init position and orientation of the robot
		Position p = simulator.getPosition("youBot");
		
		if(config.difficulty <= 1){
			x = 0;
			y = 0;
			o = 0;
		} else {
			x = (r.nextFloat()-0.5f);
			y = (r.nextFloat()-0.5f)*1.8f;
			o = (r.nextFloat()-0.5f)*6.28f;
		}
		// somehow two times setPosition was required to actually get the position set
		// TODO check in VREP simulator source?
		simulator.setPosition("youBot", new Position(x, y, p.z));
		simulator.setPosition("youBot", new Position(x, y, p.z));
		simulator.setOrientation("youBot", new Orientation(-1.5707963f, o, -1.5707965f));
		
		// set random can position
		float s = 0;
		while(s < 0.15f) { // can should not be colliding with youbot from start
			p = simulator.getPosition("Can1");
			
			if(config.difficulty == 0){
				x = 0;
			} else {
				x = (r.nextFloat()-0.5f)*1.6f;
			}
			
			if(config.difficulty <= 1){
				y = (0.125f + 3*r.nextFloat()/8f)*2.4f;
			} else {
				y = (r.nextFloat()-0.5f)*2.4f;
			}
			
			simulator.setPosition("Can1", new Position(x, y, 0.06f));
			simulator.setOrientation("Can1", new Orientation(0, 0 ,1.6230719f));
			
			Position d = simulator.getPosition("Can1", "youBot");
			s = d.y*d.y+d.z*d.z;
		} 
		
		simulator.start(true);
		
		// TODO there might be an issue with range sensor not coming online at all
		// should be fixed in robot project..
		long start = System.currentTimeMillis();
		int tries = 0;
		while(rangeSensor==null
				|| kukaArm == null 
				|| kukaPlatform == null){
			try {
				Thread.sleep(100);
				simulator.tick();
			} catch (InterruptedException|TimeoutException e) {
			}
			
			if(System.currentTimeMillis()-start > 20000){
				tries++;
				if(tries >= 3){
					throw new Exception("Failed to initialize Kuka environment");
				}
				
				System.out.println("Failed to initialize youbot/laserscanner in environment... Try again");

				// try again?
				simulator.stop();
				Thread.sleep(1000);
				simulator.start(true);
				start = System.currentTimeMillis();
			}
		}
		
		// calculate reward here to initialize previousDistance
		calculateReward();
	}
	
	
	@Override
	public void configure(Map<String, String> config) {
		// configure the simulated environment
		if(simulator != null){
			Map<String, String> entities = new HashMap<String, String>();
			entities.put("youBot", "be.iminds.iot.robot.youbot.ros.Youbot");
			entities.put("hokuyo", "be.iminds.iot.sensor.range.ros.LaserScanner");
			
			simulator.loadScene("scenes/youbot_fetch_can.ttt", entities);
		} 
		
		this.config = DianneConfigHandler.getConfig(config, FetchCanConfig.class);
	}
}
