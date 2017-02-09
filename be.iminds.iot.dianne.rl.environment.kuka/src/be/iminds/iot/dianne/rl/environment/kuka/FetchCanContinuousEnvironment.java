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
import be.iminds.iot.dianne.rl.environment.kuka.config.FetchCanContinuousConfig;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;
import be.iminds.iot.robot.api.Arm;


@Component(immediate = true,
	service = {Environment.class, KukaEnvironment.class},
	property = { "name="+FetchCanContinuousEnvironment.NAME, 
				 "aiolos.unique=true",
				 "aiolos.combine=*",
				 "osgi.command.scope=fetchcancont",
				 "osgi.command.function=start",
				 "osgi.command.function=stop",
				 "osgi.command.function=pause",
				 "osgi.command.function=resume",
				 "osgi.command.function=reward",
				 "osgi.command.function=resetCan"})
public class FetchCanContinuousEnvironment extends AbstractFetchCanEnvironment {
	
	public static final String NAME = "FetchCanContinuous";
	
	protected FetchCanContinuousConfig config;
	
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
		if(TensorOps.dot(a, a) < config.stopThreshold) {
			grip = true;
			
			kukaPlatform.stop();	

			if(!super.config.earlyStop) {
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
				if(simulator != null && super.config.tick){
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
			
			}
		} else {
			float[] action = a.get();
			kukaPlatform.move(action[0]*super.config.speed, action[1]*super.config.speed, action[2]*super.config.speed*2);
		}
	
		// simulate an iteration further
		if(simulator != null && super.config.tick){
			for(int i=0;i<=super.config.skip;i++){
				simulator.tick();
			}
		}
	}
	
	@Override
	public void configure(Map<String, String> config) {
		this.config = DianneConfigHandler.getConfig(config, FetchCanContinuousConfig.class);
		
		super.configure(config);
	}
}
