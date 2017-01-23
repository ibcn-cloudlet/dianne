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
package be.iminds.iot.dianne.things.input;

import java.util.Dictionary;
import java.util.Hashtable;
import java.util.UUID;

import org.osgi.framework.BundleContext;
import org.osgi.framework.ServiceRegistration;

import be.iminds.iot.dianne.api.nn.module.Input;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.sensor.api.SensorListener;
import be.iminds.iot.sensor.api.SensorValue;

public class LaserScanInput extends ThingInput implements SensorListener {

	private Tensor t;
	
	private ServiceRegistration registration;
	
	public LaserScanInput(UUID id, String name){
		super(id, name, "LaserScanner");
	}

	@Override
	public void update(SensorValue value) {
		try {
			for(Input in : inputs){
				if(t == null || t.size() != value.data.length){
					t = new Tensor(value.data.length);
				}
				t.set(value.data);
				in.input(t);
			}
		} catch(Exception e){
			// ignore exception ... probably NN was deployed while forwarding input
		}
	}

	@Override
	public void connect(Input input, BundleContext context) {
		super.connect(input, context);
		
		if(registration == null){
			Dictionary<String, Object> properties = new Hashtable<String, Object>();
			properties.put("target", id.toString());
			properties.put("aiolos.unique", true);
			registration = context.registerService(SensorListener.class.getName(), this, properties);
		}
	}

	
	@Override
	public void disconnect(Input input){
		super.disconnect(input);
		
		if(registration != null && inputs.size() == 0){
			registration.unregister();
			registration = null;
		}
	}
	
}
