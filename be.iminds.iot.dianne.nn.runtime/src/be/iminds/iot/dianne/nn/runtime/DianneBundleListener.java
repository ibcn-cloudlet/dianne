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
package be.iminds.iot.dianne.nn.runtime;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.IOException;
import java.net.URL;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.Map;
import java.util.UUID;

import org.osgi.framework.Bundle;
import org.osgi.framework.BundleContext;
import org.osgi.framework.BundleEvent;
import org.osgi.framework.SynchronousBundleListener;
import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Deactivate;
import org.osgi.service.component.annotations.Reference;

import be.iminds.iot.dianne.api.nn.Dianne;
import be.iminds.iot.dianne.api.nn.module.dto.ModuleInstanceDTO;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkDTO;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkInstanceDTO;
import be.iminds.iot.dianne.api.nn.runtime.DianneRuntime;
import be.iminds.iot.dianne.nn.util.DianneJSONConverter;
import be.iminds.iot.dianne.tensor.Tensor;

@Component
public class DianneBundleListener implements SynchronousBundleListener {

	private DianneRuntime runtime;
	private Dianne dianne;
	
	private Map<Bundle, NeuralNetworkInstanceDTO> nns = new HashMap<>();
	
	@Reference
	void setDianneRuntime(DianneRuntime r){
		runtime = r;
	}
	
	@Reference
	void setDianne(Dianne d){
		dianne = d;
	}
	
	@Activate
	public void activate(BundleContext context){
		context.addBundleListener(this);
		
		for(Bundle b :context.getBundles()){
			if(b.getHeaders().get("NeuralNetwork")!=null)
				nnBundleStarting(b);
		}
		
	}
	
	@Deactivate
	public void deactivate(){
		
	}
	
	@Override
	public void bundleChanged(BundleEvent event) {
		if(event.getBundle().getHeaders().get("NeuralNetwork")==null)
			return;
			
		if(event.getType()==BundleEvent.STARTING){
			nnBundleStarting(event.getBundle());
		} else if(event.getType()==BundleEvent.STOPPING){
			nnBundleStopping(event.getBundle());
		}
	}

	
	private void nnBundleStarting(Bundle b){
		try {
			URL url = b.getEntry("modules.txt");
			
			byte[] data = new byte[1000];
			StringBuilder builder = new StringBuilder();
			DataInputStream is = new DataInputStream(url.openStream());
			int read = 0;
			while( (read = is.read(data)) >= 0){
				builder.append(new String(data, 0, read));
			}
			String nn = builder.toString();
			NeuralNetworkDTO dto = DianneJSONConverter.parseJSON(nn);
		
			final UUID nnId = UUID.randomUUID();
			final Map<UUID, ModuleInstanceDTO> modules = new HashMap<>();
			dto.modules.values().stream().forEach(m -> {
				Tensor params = loadParameters(b, m.id);
				ModuleInstanceDTO mi = runtime.deployModule(m, nnId, params);
				modules.put(mi.moduleId, mi);
			});
			
			NeuralNetworkInstanceDTO nni = new NeuralNetworkInstanceDTO(nnId, dto.name, b.getSymbolicName(),modules);
			nns.put(b, nni);
			dianne.getNeuralNetwork(nni);
			
		} catch(Exception e){
			e.printStackTrace();
			System.out.println("Failed to install neural network from bundle "+b.getSymbolicName());
		}
	}
	
	private void nnBundleStopping(Bundle b){
		NeuralNetworkInstanceDTO nni = nns.get(b);
		if(nni != null){
			runtime.undeployModules(nni.id);
		}
	}
	
	private Tensor loadParameters(Bundle b, UUID moduleId){
		Tensor params = null;
		Enumeration<URL> e = b.findEntries("/", moduleId.toString()+"*", true);
		if(e != null){
			URL url = e.nextElement();
			try(DataInputStream is = new DataInputStream(new BufferedInputStream(url.openStream()))){
				params = readTensor(is);
			} catch(Exception ex){}
		}
		
		return params;
	}
	
	private Tensor readTensor(DataInputStream is) throws IOException{
		// load tensor in chunks, slightly slower than one copy from Java to native,
		// but reduces memory usage a lot for big tensors
		int bufferSize = 10000;
		float[] data = new float[bufferSize];
		
		int length = is.readInt();
		Tensor t = new Tensor(length);
		int index = 0;
		while(length > 0){
			if(length<bufferSize){
				bufferSize = length;
				data = new float[bufferSize];
			}
			
			for(int i=0;i<bufferSize;i++){
				data[i] = is.readFloat();
			}
			
			t.narrow(0, index, bufferSize).set(data);;
			
			length -= bufferSize;
			index+= bufferSize;
		}
		return t;
	}
}
