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
import java.util.Dictionary;
import java.util.HashMap;
import java.util.Hashtable;
import java.util.List;
import java.util.Map;
import java.util.UUID;

import org.osgi.framework.BundleContext;
import org.osgi.framework.ServiceRegistration;
import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;
import org.osgi.service.component.annotations.ReferenceCardinality;
import org.osgi.service.component.annotations.ReferencePolicy;

import be.iminds.iot.dianne.api.io.OutputDescription;
import be.iminds.iot.dianne.api.io.DianneOutputs;
import be.iminds.iot.dianne.api.nn.module.ForwardListener;
import be.iminds.iot.dianne.tensor.TensorFactory;
import be.iminds.iot.things.api.Thing;
import be.iminds.iot.things.api.lamp.Lamp;


@Component(immediate=true)
public class ThingsOutputs implements DianneOutputs {

	private BundleContext context;
	
	private TensorFactory factory;
	
	private Map<String, Lamp> lamps = Collections.synchronizedMap(new HashMap<String, Lamp>());
	
	private Map<UUID, ServiceRegistration> registrations =  Collections.synchronizedMap(new HashMap<UUID, ServiceRegistration>());
	
	private int magicNumber = -1;
	
	@Activate
	void activate(BundleContext context){
		this.context = context;
		String number = context.getProperty("be.iminds.iot.dianne.things.light.index");
		if(number!=null){
			magicNumber = Integer.parseInt(number);
		}
	}
	
	@Reference
	void setTensorFactory(TensorFactory factory){
		this.factory = factory;
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
		
		// TODO only handle local thing services?
		// TODO use id-service combo?
		lamps.put(service, l);
	}
	
	void removeLamp(Lamp l, Map<String, Object> properties){
		UUID id = UUID.fromString((String)properties.get(Thing.ID));
		String service = (String) properties.get(Thing.SERVICE);
		lamps.remove(service);
	}

	@Override
	public List<OutputDescription> getAvailableOutputs() {
		ArrayList<OutputDescription> outputs = new ArrayList<OutputDescription>();
		synchronized(lamps){
			for(String id : lamps.keySet()){
				outputs.add(new OutputDescription(id, "Lamp"));
			}
		}
		return outputs;
	}

	@Override
	public void setOutput(UUID nnId, UUID outputId, String output) {
		String id = nnId.toString()+":"+outputId.toString();

		Lamp l = lamps.get(output);
		if(l!=null){
			LampOutput o = new LampOutput(factory, l, magicNumber);
			Dictionary<String, Object> properties = new Hashtable<String, Object>();
			properties.put("targets", new String[]{id});
			properties.put("aiolos.unique", true);
			ServiceRegistration r = context.registerService(ForwardListener.class.getName(), o, properties);
			// TODO only works if outputId only forwards to one output
			registrations.put(outputId, r);
		}
	}

	@Override
	public void unsetOutput(UUID nnId, UUID outputId, String output) {
		String id = nnId.toString()+":"+outputId.toString();

		ServiceRegistration r = registrations.remove(id);
		if(r!=null){
			r.unregister();
		}
	}

}
