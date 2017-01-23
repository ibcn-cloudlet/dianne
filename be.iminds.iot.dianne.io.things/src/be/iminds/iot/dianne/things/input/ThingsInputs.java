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

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.UUID;

import org.osgi.framework.BundleContext;
import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;
import org.osgi.service.component.annotations.ReferenceCardinality;
import org.osgi.service.component.annotations.ReferencePolicy;

import be.iminds.iot.dianne.api.io.DianneInputs;
import be.iminds.iot.dianne.api.io.InputDescription;
import be.iminds.iot.dianne.api.nn.module.Input;
import be.iminds.iot.sensor.api.LaserScanner;
import be.iminds.iot.things.api.Thing;
import be.iminds.iot.things.api.camera.Camera;

@Component(immediate=true,
	property = {"aiolos.unique=true"})
public class ThingsInputs implements DianneInputs {

	private BundleContext context;

	// these are mapped by the string nnId:moduleId  ... TODO use ModuleInstanceDTO for this?
	private Map<String, Input> inputs = Collections.synchronizedMap(new HashMap<String, Input>());
	// things mapped by uuid
	private Map<UUID, ThingInput> things = Collections.synchronizedMap(new HashMap<>());
	
	@Activate
	void activate(BundleContext context){
		this.context = context;
	}
	
	@Reference(
			cardinality=ReferenceCardinality.MULTIPLE, 
			policy=ReferencePolicy.DYNAMIC)
	void addCamera(Camera c, Map<String, Object> properties){
		UUID id = UUID.fromString((String)properties.get(Thing.ID));
		String service = (String) properties.get(Thing.SERVICE);
		
		String[] parts = service.split("_");
		String filtered = "Webcam "+parts[1];
		// TODO only handle local thing services?
		// TODO use id-service combo?
		CameraInput camera = new CameraInput(id, filtered, c);
		things.put(id, camera);
		
	}
	
	void removeCamera(Camera c, Map<String, Object> properties){
		UUID id = UUID.fromString((String)properties.get(Thing.ID));
		things.remove(id);
	}

	
	@Reference(
			cardinality=ReferenceCardinality.MULTIPLE, 
			policy=ReferencePolicy.DYNAMIC)
	void addLaserScanner(LaserScanner l, Map<String, Object> properties){
		String name = (String) properties.get("name");
		UUID id = UUID.nameUUIDFromBytes(name.getBytes());
		
		String cap = name.substring(0, 1).toUpperCase() + name.substring(1).toLowerCase();
		LaserScanInput laser = new LaserScanInput(id, cap);
		things.put(id, laser);
	}
	
	void removeLaserScanner(LaserScanner l, Map<String, Object> properties){
		String name = (String) properties.get("name");
		UUID id = UUID.nameUUIDFromBytes(name.getBytes());		
		things.remove(id);
	}
	
	@Reference(cardinality=ReferenceCardinality.MULTIPLE, 
			policy=ReferencePolicy.DYNAMIC)
	void addInput(Input i, Map<String, Object> properties){
		String moduleId = (String)properties.get("module.id");
		String nnId = (String)properties.get("nn.id");
		String id = nnId+":"+moduleId;
		inputs.put(id, i);
	}
	
	void removeInput(Input i, Map<String, Object> properties){
		String moduleId = (String)properties.get("module.id");
		String nnId = (String)properties.get("nn.id");
		String id = nnId+":"+moduleId;
		Input in = inputs.remove(id);
		synchronized(things){
			for(ThingInput t : things.values()){
				t.disconnect(in);
			}
		}
	}
	
	@Override
	public List<InputDescription> getAvailableInputs() {
		ArrayList<InputDescription> inputs = new ArrayList<InputDescription>();
		synchronized(things){
			for(ThingInput t : things.values()){
				inputs.add(t.getInputDescription());
			}
		}
		return inputs;
	}

	@Override
	public void setInput(UUID nnId, UUID inputId, String input) {
		String id = nnId.toString()+":"+inputId.toString();
		Input in = inputs.get(id);
		ThingInput thing = null;
		try {
			synchronized(things){
				thing = things.values().stream().filter(t -> t.name.equals(input)).findFirst().get();
			}
		} catch(NoSuchElementException e){
			return;
		}
		thing.connect(in, context);
	}

	@Override
	public void unsetInput(UUID nnId, UUID inputId, String input) {
		String id = nnId.toString()+":"+inputId.toString();
		Input in = inputs.get(id);
		ThingInput thing = null;
		try {
			synchronized(things){
				thing = things.values().stream().filter(t -> t.name.equals(input)).findFirst().get();
			}
		} catch(NoSuchElementException e){
			return;
		}
		thing.disconnect(in);
	}

}
