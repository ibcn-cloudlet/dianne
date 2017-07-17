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
import java.util.concurrent.TimeoutException;

import org.osgi.service.component.annotations.Component;
import org.osgi.util.promise.Promise;

import be.iminds.iot.dianne.api.rl.environment.Environment;
import be.iminds.iot.dianne.nn.util.DianneConfigHandler;
import be.iminds.iot.dianne.rl.environment.kuka.api.KukaEnvironment;
import be.iminds.iot.dianne.rl.environment.kuka.config.FetchCanReacherIKConfig;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.robot.api.arm.Arm;
import be.iminds.iot.simulator.api.Orientation;
import be.iminds.iot.simulator.api.Position;


@Component(immediate = true,
	service = {Environment.class, KukaEnvironment.class},
	property = { "name="+FetchCanReacherIKEnvironment.NAME, 
				 "aiolos.unique=true",
				 "aiolos.combine=*",
				 "osgi.command.scope=reacherIK",
				 "osgi.command.function=start",
				 "osgi.command.function=stop",
				 "osgi.command.function=pause",
				 "osgi.command.function=resume",
				 "osgi.command.function=reward"})
public class FetchCanReacherIKEnvironment extends AbstractFetchCanEnvironment {
	
	public static final String NAME = "FetchCanReacherIK";
	
	protected FetchCanReacherIKConfig config;
	
	@Override
	public String getName(){
		return NAME;
	}

	@Override
	public int[] actionDims() {
		return new int[]{3};
	}
	
	@Override
	public void configure(Map<String, String> config) {
		this.config = DianneConfigHandler.getConfig(config, FetchCanReacherIKConfig.class);
		
		super.configure(config);
	}
	
	@Override
	protected void executeAction(Tensor a) throws Exception {
		float x,y;
		if(config.baseline && simulator != null){
			Position p = simulator.getPosition("can_ref", "arm_ref");
			x = p.x;
			y = p.y;
		} else {
			x = a.get(0);
			y = a.get(1);
		}

		Promise<Arm> p = kukaArm.moveTo(x, y, config.hoverHeight)
			.then(pp -> kukaArm.moveTo(x, y, config.gripHeight))
			.then(pp -> kukaArm.closeGripper())
			.then(pp -> kukaArm.moveTo(0.4f, 0.0f, 0.4f));
		
		// simulate an iteration further
		int i = 0;
		if(simulator != null && super.config.tick){
			while(!p.isDone() && i++ < 100){
				simulator.tick();
			}
		}
		
		// only one try ?
		terminal = true;
	}
	
	@Override
	protected float calculateReward() throws Exception {
		// calculate reward based on simulator info
		if(simulator != null){
			if(simulator.getPosition("can_ref").z > 0.2f){
				return 1;
			} else {
				return 0;
			}
		} 
		
		// in case no simulator ... return reward variable that might be set manually
		return reward;
	}

	protected void resetEnvironment(){
		// youbot x,y and orientation
		float x,y,o;
		// can x,y
		float cx,cy;
		
		if(super.config.difficulty <= 0){
			// fix youbot at 0,0
			x = 0;
			y = 0;
			o = 0;
		} else if(super.config.difficulty == 1){
			// random youbot position and orientation (ensure it fits in boundaries)
			x = (r.nextFloat()-0.5f)*0.6f;
			y = (r.nextFloat()-0.5f)*1.5f;
			o = (r.nextFloat()-0.5f)*6.28f;
		} else {
			// no assumptions made on can being graspable
			x = (r.nextFloat()-0.5f);
			y = (r.nextFloat()-0.5f)*1.8f;
			o = (r.nextFloat()-0.5f)*6.28f;
		}

		float d;
		double a;
		if(super.config.difficulty <= 1){
			// make sure can is in workspace of robot
			d = r.nextFloat()*0.25f;
			a = (r.nextFloat()-0.5f)*Math.PI;
		} else {
			// can might be just to far out
			d = r.nextFloat()*0.5f;
			a = (r.nextFloat()-0.5f)*Math.PI;
		}
		
		cx = x + (float)Math.sin(o)*0.4f + (float)Math.sin(o+a)*d;
		cy = y + (float)Math.cos(o)*0.4f + (float)Math.cos(o+a)*d;
		
		simulator.setPosition("youBot", new Position(x, y, 0.0957f));
		simulator.setOrientation("youBot", new Orientation(-1.5707963f, o, -1.5707965f));
		
		simulator.setOrientation("Can1", new Orientation(0, 0 ,1.6230719f));
		simulator.setPosition("Can1", new Position(cx, cy, 0.06f));
	}
	
	
	@Override
	protected void initAction(){
		// reset arm to candle
		Promise<Arm> p = kukaArm.moveTo(0.4f, 0.0f, 0.4f);
		int i = 0;
		while(!p.isDone() && active && i < 100) {
			if (super.config.tick) {
				try {
					simulator.tick();
					i++;
				} catch(TimeoutException e){}
			} else {
				try {
					Thread.sleep(100);
				} catch (InterruptedException e) {
				}
			}
		}
	}
}
