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
package be.iminds.iot.dianne.things.output;

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

import be.iminds.iot.dianne.api.io.DianneOutputs;
import be.iminds.iot.dianne.api.io.OutputDescription;
import be.iminds.iot.robot.api.Arm;
import be.iminds.iot.robot.api.OmniDirectional;
import be.iminds.iot.things.api.Thing;
import be.iminds.iot.things.api.lamp.Lamp;


@Component(immediate=true,
	property = {"aiolos.unique=true"})
public class ThingsOutputs implements DianneOutputs {

	private BundleContext context;
	
	private Map<UUID, ThingOutput> things = Collections.synchronizedMap(new HashMap<>());
	
	@Activate
	void activate(BundleContext context){
		this.context = context;
	}
	
	@Reference(
			cardinality=ReferenceCardinality.MULTIPLE, 
			policy=ReferencePolicy.DYNAMIC)
	void addLamp(Lamp l, Map<String, Object> properties){
		UUID id = UUID.fromString((String)properties.get(Thing.ID));
		String service = (String) properties.get(Thing.SERVICE);
		
		// TODO hard coded Hue for now...
		if(service.startsWith("philips")){
			service = "Philips Hue";
		}
		
		things.put(id, new LampOutput(id, service, l));
	}
	
	void removeLamp(Lamp l, Map<String, Object> properties){
		UUID id = UUID.fromString((String)properties.get(Thing.ID));
		ThingOutput t = things.remove(id);
		if(t != null){
			t.disconnect();
		}
	}
	
	@Reference(
			cardinality=ReferenceCardinality.MULTIPLE, 
			policy=ReferencePolicy.DYNAMIC)
	void addBase(OmniDirectional b, Map<String, Object> properties){
		String name = (String) properties.get("name");
		UUID id = UUID.nameUUIDFromBytes(name.getBytes());
		
		synchronized(things){
			ThingOutput t = things.get(id);
			if(t == null){
				t = new YoubotOutput(id, name, context);
				things.put(id, t);
			}
			((YoubotOutput)t).setBase(b);
		}
	}
	
	void removeBase(OmniDirectional b, Map<String, Object> properties){
		String name = (String) properties.get("name");
		UUID id = UUID.nameUUIDFromBytes(name.getBytes());
		ThingOutput t = things.remove(id);
		if(t != null){
			t.disconnect();
		}
	}

	@Reference(
			cardinality=ReferenceCardinality.MULTIPLE, 
			policy=ReferencePolicy.DYNAMIC)
	void addArm(Arm a, Map<String, Object> properties){
		String name = (String) properties.get("name");
		UUID id = UUID.nameUUIDFromBytes(name.getBytes());
		
		synchronized(things){
			ThingOutput t = things.get(id);
			if(t == null){
				t = new YoubotOutput(id, name, context);
				things.put(id, t);
			}
			((YoubotOutput)t).setArm(a);
		}
	}
	
	void removeArm(Arm a, Map<String, Object> properties){
		String name = (String) properties.get("name");
		UUID id = UUID.nameUUIDFromBytes(name.getBytes());
		ThingOutput t = things.remove(id);
		if(t != null){
			t.disconnect();
		}
	}
	
	@Override
	public List<OutputDescription> getAvailableOutputs() {
		ArrayList<OutputDescription> outputs = new ArrayList<OutputDescription>();
		synchronized(things){
			for(ThingOutput t : things.values()){
				outputs.add(t.getOutputDescription());
			}
		}
		return outputs;
	}

	@Override
	public void setOutput(UUID nnId, UUID outputId, String output) {
		ThingOutput o = null;
		try {
			synchronized(things){
				o = things.values().stream().filter(t -> t.name.equals(output)).findFirst().get();
			}
		} catch(NoSuchElementException e){
			return;
		}
		o.connect(nnId, outputId, context);
	}

	@Override
	public void unsetOutput(UUID nnId, UUID outputId, String output) {
		ThingOutput o = null;
		try {
			synchronized(things){
				o = things.values().stream().filter(t -> t.name.equals(output)).findFirst().get();
			}
		} catch(NoSuchElementException e){
			return;
		}
		o.disconnect(nnId, outputId);
	}

}
