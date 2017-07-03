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
import org.osgi.util.promise.Promise;

import be.iminds.iot.dianne.api.rl.environment.Environment;
import be.iminds.iot.dianne.rl.environment.kuka.api.KukaEnvironment;
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
	
	@Override
	public String getName(){
		return NAME;
	}

	@Override
	public int[] actionDims() {
		return new int[]{3};
	}
	
	@Override
	protected void executeAction(Tensor a) throws Exception {
		float[] xyz = a.get();
		
		Promise<Arm> p = kukaArm.moveTo(xyz[0], xyz[1], 0.1f)
			.then(pp -> kukaArm.moveTo(xyz[0], xyz[1], 0.05f))
			.then(pp -> kukaArm.closeGripper())
			.then(pp -> kukaArm.moveTo(xyz[0], xyz[1], 0.2f));
		
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
			if(checkCollisions()){
				return -1;
			} else if(simulator.getPosition("can_ref").z > 0.1f){
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
		} else {
			// random youbot position and orientation
			x = (r.nextFloat()-0.5f)*0.7f;
			y = (r.nextFloat()-0.5f)*1.5f;
			o = (r.nextFloat()-0.5f)*6.28f;
		}

		// put can in workspace of robot
		float d = r.nextFloat()*0.25f;
		double a = (r.nextFloat()-0.5f)*Math.PI;
		
		cx = x + (float)Math.sin(o)*0.4f + (float)Math.sin(o+a)*d;
		cy = y + (float)Math.cos(o)*0.4f + (float)Math.cos(o+a)*d;
		
		simulator.setPosition("youBot", new Position(x, y, 0.0957f));
		simulator.setOrientation("youBot", new Orientation(-1.5707963f, o, -1.5707965f));
		
		simulator.setOrientation("Can1", new Orientation(0, 0 ,1.6230719f));
		simulator.setPosition("Can1", new Position(cx, cy, 0.06f));
	}
	
}
