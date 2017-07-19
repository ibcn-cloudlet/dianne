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


@Component(immediate = true,
	service = {Environment.class, KukaEnvironment.class},
	property = { "name="+DockingContinuousEnvironment.NAME, 
				 "aiolos.unique=true",
				 "aiolos.combine=*",
				 "osgi.command.scope=dockingcont",
				 "osgi.command.function=start",
				 "osgi.command.function=stop",
				 "osgi.command.function=pause",
				 "osgi.command.function=resume",
				 "osgi.command.function=reward",
				 "osgi.command.function=resetCan"})
public class DockingContinuousEnvironment extends DockingEnvironment {
	
	public static final String NAME = "DockingContinuous";
	
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
		float[] velocities = a.get();
		kukaPlatform.move(velocities[0]*super.config.speed, velocities[1]*super.config.speed, velocities[2]*super.config.speed*2);
		
		// simulate an iteration further
		if(simulator != null){
			for(int i=0;i<=super.config.skip;i++){
				simulator.tick();
			}
		}
	}
}
