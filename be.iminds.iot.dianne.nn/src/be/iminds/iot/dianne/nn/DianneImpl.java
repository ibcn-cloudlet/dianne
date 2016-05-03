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
package be.iminds.iot.dianne.nn;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.UUID;

import org.osgi.framework.BundleContext;
import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;
import org.osgi.service.component.annotations.ReferenceCardinality;
import org.osgi.service.component.annotations.ReferencePolicy;
import org.osgi.util.promise.Deferred;
import org.osgi.util.promise.Promise;

import be.iminds.iot.dianne.api.nn.Dianne;
import be.iminds.iot.dianne.api.nn.NeuralNetwork;
import be.iminds.iot.dianne.api.nn.module.Module;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkInstanceDTO;
import be.iminds.iot.dianne.api.repository.DianneRepository;

@Component(property={"aiolos.export=false"})
public class DianneImpl implements Dianne {

	private BundleContext context;
	
	private DianneRepository repository;
	
	private Map<UUID, Map<UUID, Module>> modules = Collections.synchronizedMap(new HashMap<>());
	
	private Map<UUID, NeuralNetworkWrapper> nns = Collections.synchronizedMap(new HashMap<UUID, NeuralNetworkWrapper>());
	private Map<UUID, NeuralNetwork> nnServices = Collections.synchronizedMap(new HashMap<UUID, NeuralNetwork>());
	
	// a set of nnIds to watch for and bring online once all modules are there
	private Map<UUID, ToWatch> toWatchFor = Collections.synchronizedMap(new HashMap<>());
	
	private class ToWatch {
		NeuralNetworkInstanceDTO nni;
		Deferred<NeuralNetwork> deferred;
	}
	
	@Override
	public Promise<NeuralNetwork> getNeuralNetwork(NeuralNetworkInstanceDTO nni) {
		Deferred<NeuralNetwork> result = new Deferred<>();
		
		NeuralNetwork nn = nnServices.get(nni.id);
		if(nn!=null){
			result.resolve(nn);
			return result.getPromise();
		}
		
		ToWatch w = new ToWatch();
		w.nni = nni;
		w.deferred = result; 
		toWatchFor.put(nni.id, w);
		
		createNeuralNetwork(nni.id);
		
		return result.getPromise();
	}

	private void createNeuralNetwork(UUID nnId){
		// try to create new NeuralNetworkWrapper for this instance
		if(!toWatchFor.containsKey(nnId))
			return;
		
		NeuralNetworkInstanceDTO nni = toWatchFor.get(nnId).nni;
		
		Map<UUID, Module> m = modules.get(nnId);
		if(m.size() == nni.modules.size()){
			// all modules available
			NeuralNetworkWrapper wrapper = new NeuralNetworkWrapper(nni, m.values(), repository, context);
			wrapper.register();
			nns.put(nni.id, wrapper);
		} else {
			System.out.println("Tried to create service for NN "+nnId+", but not all modules are present.");
		}
	}
	
	@Activate
	public void activate(BundleContext ctx){
		this.context = ctx;
	}
	
	@Reference
	void setDianneRepository(DianneRepository r){
		repository = r;
	}
	
	@Reference(cardinality=ReferenceCardinality.MULTIPLE, 
			policy=ReferencePolicy.DYNAMIC)
	void addModule(Module m, Map<String, Object> properties){
		UUID moduleId = UUID.fromString((String)properties.get("module.id"));
		UUID nnId = UUID.fromString((String)properties.get("nn.id"));
	
		synchronized(modules){
			Map<UUID, Module> nnm = modules.get(nnId);
			if(nnm==null){
				nnm = new HashMap<>();
				modules.put(nnId, nnm);
			}
			nnm.put(moduleId, m);
		}
		
		createNeuralNetwork(nnId);
	}
	
	void removeModule(Module m, Map<String, Object> properties){
		UUID moduleId = UUID.fromString((String)properties.get("module.id"));
		UUID nnId = UUID.fromString((String)properties.get("nn.id"));

		synchronized(modules){
			Map<UUID, Module> nnm = modules.get(nnId);
			if(nnm != null){ // should not be null?
				nnm.remove(moduleId);
				
				if(nnm.size() == 0){
					modules.remove(nnId);
				}
			}
		}
		
		// a module is removed, tear down NeuralNetwork service
		NeuralNetworkWrapper nn = nns.remove(nnId);
		if(nn!=null)
			nn.unregister();
		
	}
	
	@Reference(cardinality=ReferenceCardinality.MULTIPLE, 
			policy=ReferencePolicy.DYNAMIC)
	public void addNeuralNetwork(NeuralNetwork nn, Map<String, Object> properties){
		String nnId = (String)properties.get("nn.id");
		UUID id = UUID.fromString(nnId);
		nnServices.put(id, nn);
		
		ToWatch watched = toWatchFor.remove(id);
		if(watched!=null){
			watched.deferred.resolve(nn);
		}
	}
	
	public void removeNeuralNetwork(NeuralNetwork nn, Map<String, Object> properties){
		String nnId = (String)properties.get("nn.id");
		nnServices.remove(UUID.fromString(nnId));
	}
	
}
