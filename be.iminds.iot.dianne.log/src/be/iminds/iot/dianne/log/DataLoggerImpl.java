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
package be.iminds.iot.dianne.log;

import java.util.HashMap;
import java.util.Map;

import org.osgi.framework.BundleContext;
import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Component;

import be.iminds.iot.dianne.api.log.DataLogger;

@Component
public class DataLoggerImpl implements DataLogger {

	private float defaultAlpha = 0.01f;
	private int defaultInterval = 1000;
	
	private Map<String, Float> averages = new HashMap<>();
	private Map<String, Integer> counts = new HashMap<>();
	
	private Map<String, Integer> intervals = new HashMap<>();
	private Map<String, Float> alphas = new HashMap<>();
	
	
	@Activate
	public void activate(BundleContext context){
		String a = context.getProperty("be.iminds.iot.dianne.log.alpha");
		if (a != null)
			this.defaultAlpha = Float.parseFloat(a);

		String i = context.getProperty("be.iminds.iot.dianne.log.interval");
		if (i != null)
			this.defaultInterval = Integer.parseInt(i);
	}
	
	@Override
	public void log(String label, String[] keys, float... values) {
		assert keys.length==values.length;
		for(int i=0;i<keys.length;i++){
			Float old = averages.get(keys[i]);
			if(old==null){
				averages.put(keys[i], values[i]);
			} else {
				Float alpha = alphas.get(keys[i]);
				if(alpha==null){
					alpha = defaultAlpha;
				}
				
				averages.put(keys[i], (1-alpha) * old  + alpha * values[i]);
			}
		}
		
		Integer interval = intervals.get(label);
		if(interval==null){
			interval = defaultInterval;
		}
		
		// count
		Integer count = counts.get(label);
		if(count==null){
			count = 0;
		} 
		count = count + 1;
		counts.put(label, count);
		
		if(interval.intValue() < 1){
			// print values directly
			StringBuilder b = new StringBuilder();
			b.append("[").append(label).append("]\t");
			b.append(count).append("\t");
			for(int i=0;i<keys.length;i++){
				b.append(keys[i]).append("\t").append(values[i]).append("\t");
			}
			System.out.println(b.toString());
			
		} else if(count.intValue() % interval.intValue() == 0){
			// print averages and reset count
			StringBuilder b = new StringBuilder();
			b.append("[").append(label).append("]\t");
			b.append(count).append("\t");
			for(int i=0;i<keys.length;i++){
				b.append(keys[i]).append("\t").append(averages.get(keys[i])).append("\t");
			}
			System.out.println(b.toString());
		}
			
	}

	@Override
	public void setInterval(String label, int interval) {
		intervals.put(label, interval);
	}

	@Override
	public void setAlpha(String key, float alpha) {
		alphas.put(key, alpha);
	}
}
