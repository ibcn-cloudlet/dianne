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
package be.iminds.iot.dianne.command;

import java.util.HashMap;
import java.util.Map;

import org.apache.felix.service.command.Descriptor;
import org.osgi.framework.BundleContext;
import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;
import org.osgi.service.component.annotations.ReferenceCardinality;
import org.osgi.service.component.annotations.ReferencePolicy;

import be.iminds.iot.dianne.api.rl.environment.Environment;

@Component(
		service=Object.class,
		property={"osgi.command.scope=dianne",
				  "osgi.command.function=environments"},
		immediate=true)
public class DianneEnvironmentCommands {

	private Map<String, Environment> environments = new HashMap<>();
	
	@Activate
	public void activate(BundleContext context){
	}
	
	@Descriptor("List the available environments.")
	public void environments(){
		if(environments.size()==0){
			System.out.println("No environments available");
			return;
		}
		
		System.out.println("Available environments:");
		int i = 0;
		for(String environment : environments.keySet()){
			System.out.println("["+(i++)+"] "+environment);
		}
	}
	
	@Reference(cardinality=ReferenceCardinality.MULTIPLE, 
			policy=ReferencePolicy.DYNAMIC)
	void addEnvironment(Environment env, Map<String, Object> properties){
		String name = (String) properties.get("name");
		this.environments.put(name, env);
	}
	
	void removeEnvironment(Environment env, Map<String, Object> properties){
		String name = (String) properties.get("name");
		this.environments.remove(name);
	}
}
