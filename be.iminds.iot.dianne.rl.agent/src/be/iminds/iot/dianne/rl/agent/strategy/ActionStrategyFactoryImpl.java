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
package be.iminds.iot.dianne.rl.agent.strategy;

import java.net.URL;
import java.util.List;

import org.osgi.framework.BundleContext;
import org.osgi.framework.wiring.BundleWiring;
import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Component;

import be.iminds.iot.dianne.api.rl.agent.ActionStrategy;
import be.iminds.iot.dianne.api.rl.agent.ActionStrategyFactory;

@Component
public class ActionStrategyFactoryImpl implements ActionStrategyFactory {

	private BundleWiring wiring;
	
	@Activate
	void activate(BundleContext context){
		wiring = context.getBundle().adapt(BundleWiring.class);
	}
	
	public ActionStrategy createActionStrategy(String name){
		Class c;
		try {
			if(name.contains(".")){
				// fully qualified class name, load directly
				c = wiring.getClassLoader().loadClass(name);

			} else {
				// search resource in this bundles wiring
				List<URL> urls = wiring.findEntries("/", name+".class", BundleWiring.FINDENTRIES_RECURSE);
				
				if(urls.size() == 0){
					System.err.println("LearningStrategy "+name+" not found");
					return null;
				}
				
				String u = urls.get(0).toString().substring(9);
				u = u.substring(u.indexOf("/")+1, u.length()-6);
				u = u.replaceAll("/", ".");
		
				c = this.getClass().getClassLoader().loadClass(u);
			}
			return (ActionStrategy) c.newInstance();
		} catch (Throwable e) {
			e.printStackTrace();
		}
		
		return null;
	}
	
}
