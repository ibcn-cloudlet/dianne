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
import java.util.HashMap;
import java.util.Hashtable;
import java.util.Map;
import java.util.UUID;

import org.osgi.framework.BundleContext;
import org.osgi.framework.ServiceRegistration;

import be.iminds.iot.dianne.api.io.InputDescription;
import be.iminds.iot.dianne.api.nn.module.Input;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.sensor.api.Camera;
import be.iminds.iot.sensor.api.Frame;
import be.iminds.iot.sensor.api.LaserScanner;
import be.iminds.iot.sensor.api.PointCloud;
import be.iminds.iot.sensor.api.Scanner3D;
import be.iminds.iot.sensor.api.Sensor;
import be.iminds.iot.sensor.api.SensorListener;
import be.iminds.iot.sensor.api.SensorValue;

public class SensorInput extends ThingInput implements SensorListener {

	private Sensor sensor;
	private Tensor t;
	
	private ServiceRegistration registration;
	
	public SensorInput(UUID id, String name, Sensor s){
		super(id, name, "Sensor");
		if(s instanceof LaserScanner) {
			super.type = "LaserScanner";
		} else if(s instanceof Camera) {
			super.type = "Camera";
		} else if(s instanceof Scanner3D) {
			super.type = "Scanner3D";
		}
		this.sensor = s;
	}

	@Override
	public InputDescription getInputDescription(){
		Map<String, String> properties = new HashMap<>();
		if(sensor instanceof LaserScanner) {
			LaserScanner laser = (LaserScanner) sensor;
			properties.put("minAngle", ""+laser.getMinAngle());
			properties.put("maxAngle", ""+laser.getMaxAngle());
		} else if(sensor instanceof Camera) {
			Camera camera = (Camera) sensor;
			properties.put("width", ""+camera.getWidth());
			properties.put("height", ""+camera.getHeight());
			properties.put("encoding", ""+camera.getEncoding());
		} else if(sensor instanceof Scanner3D) {
		}

		return new InputDescription(name, type, properties);
	}
	
	@Override
	public void update(SensorValue value) {
		try {
			for(Input in : inputs){
				if(t == null || t.size() != value.data.length){
					t = new Tensor(value.data.length);
				}
				t.set(value.data);
				
				// adapt shape when needed
				if(type.equals("Camera")) {
					Frame f = (Frame) value;
					if(f.encoding == Frame.Encoding.RGB) {
						t.reshape(3, f.height, f.width);
					}
				} else if(type.equals("Scanner3D")) {
					PointCloud pcl = (PointCloud) value;
					t.reshape(pcl.size, pcl.fields.length);
				}
				in.input(t);
			}
		} catch(Exception e){
			// ignore exception ... probably NN was deployed while forwarding input
			e.printStackTrace();
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
