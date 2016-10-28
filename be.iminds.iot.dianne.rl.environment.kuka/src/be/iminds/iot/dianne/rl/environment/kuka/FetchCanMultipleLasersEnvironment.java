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

import be.iminds.iot.dianne.api.rl.environment.Environment;
import be.iminds.iot.dianne.nn.util.DianneConfigHandler;
import be.iminds.iot.dianne.rl.environment.kuka.api.KukaEnvironment;
import be.iminds.iot.dianne.rl.environment.kuka.config.FetchCanConfig;


@Component(immediate = true,
	service = {Environment.class, KukaEnvironment.class},
	property = { "name="+FetchCanMultipleLasersEnvironment.NAME, 
				 "aiolos.unique=true",
				 "aiolos.combine=*",
				 "osgi.command.scope=fetchcanmul",
				 "osgi.command.function=start",
				 "osgi.command.function=stop",
				 "osgi.command.function=pause",
				 "osgi.command.function=resume",
				 "osgi.command.function=reward"})
public class FetchCanMultipleLasersEnvironment extends FetchCanEnvironment {
	
	public static final String NAME = "FetchCanMultipleLasers";
		
	@Override
	public String getName(){
		return NAME;
	}
	
	@Override
	public void configure(Map<String, String> config) {		
		// configure the simulated environment
		if(simulator != null){
			Map<String, String> entities = new HashMap<String, String>();
			entities.put("youBot", "be.iminds.iot.robot.youbot.ros.Youbot");
			entities.put("hokuyo", "be.iminds.iot.sensor.range.ros.LaserScanner");
			entities.put("hokuyo#0", "be.iminds.iot.sensor.range.ros.LaserScanner");
			
			simulator.loadScene("scenes/youbot_fetch_can_multiple_lasers.ttt", entities);
			
		}
		
		this.config = DianneConfigHandler.getConfig(config, FetchCanConfig.class);
	}
}
