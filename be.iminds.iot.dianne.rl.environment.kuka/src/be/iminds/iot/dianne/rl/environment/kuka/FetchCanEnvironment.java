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

import org.osgi.service.component.annotations.Component;
import org.osgi.util.promise.Promise;

import be.iminds.iot.dianne.api.rl.environment.Environment;
import be.iminds.iot.dianne.rl.environment.kuka.api.KukaEnvironment;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;
import be.iminds.iot.robot.api.Arm;
import be.iminds.iot.simulator.api.Position;


@Component(immediate = true,
	service = {Environment.class, KukaEnvironment.class},
	property = { "name="+FetchCanEnvironment.NAME, 
				 "aiolos.unique=true",
				 "aiolos.combine=*",
				 "osgi.command.scope=fetchcan",
				 "osgi.command.function=rest",
				 "osgi.command.function=go",
				 "osgi.command.function=reward",
				 "osgi.command.function=load"})
public class FetchCanEnvironment extends AbstractFetchCanEnvironment {
	
	public static final String NAME = "FetchCan";
	
	private float speed = 0.1f;
	
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
			kukaPlatform.move(0f, speed, 0f);
			break;
		case 1:
			kukaPlatform.move(0f, -speed, 0f);
			break;
		case 2:
			kukaPlatform.move(speed, 0f, 0f);
			break;
		case 3:
			kukaPlatform.move(-speed, 0f, 0f);
			break;
		case 4:
			kukaPlatform.move(0f, 0.f, 2*speed);
			break;
		case 5:
			kukaPlatform.move(0f, 0.f, -2*speed);
			break;	
		case 6:
			terminal = true;
			
			kukaPlatform.stop();	

			// stop early in simulator when we are nowhere 
			// near the pick location (further than 10cm)
			if(simulator != null){
				Position d = simulator.getPosition("Can1", "youBot");
				if(Math.abs(d.y) > 0.1 || Math.abs(d.z - 0.58) > 0.1){
					return;
				}
			}
			
			Promise<Arm> result = kukaArm.openGripper()
				.then(p -> kukaArm.setPosition(0, 2.92f))
				.then(p -> kukaArm.setPosition(4, 2.875f))
				.then(p -> kukaArm.setPositions(2.92f, 1.76f, -1.37f, 2.55f))
				.then(p -> kukaArm.closeGripper())
				.then(p -> kukaArm.setPositions(0.01f, 0.8f))
				.then(p -> kukaArm.setPositions(0.01f, 0.8f, -1f, 2.9f))
				.then(p -> kukaArm.openGripper())
				.then(p -> kukaArm.setPosition(1, -1.3f))
				.then(p -> kukaArm.reset());
			
			
			// in simulation keep on ticking to let the action complete
			if(simulator != null){
				for(int i=0;i<300;i++){
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
			
			return;
		}
		
		// simulate an iteration further
		if(simulator != null)
			simulator.tick();	
	
	}
	
	@Override
	public void configure(Map<String, String> config) {
		
		if(config.containsKey("speed")){
			this.speed = Float.parseFloat(config.get("speed"));
		}
		
		super.configure(config);
	}
	
	@Override
	public void load(){
		configure(new HashMap<>());
	}
}
