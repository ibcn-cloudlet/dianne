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
package be.iminds.iot.dianne.nn.module.composite;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.net.URL;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.UUID;
import java.util.stream.Collectors;

import org.osgi.framework.Bundle;
import org.osgi.framework.BundleContext;
import org.osgi.framework.BundleEvent;
import org.osgi.framework.BundleListener;
import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;

import be.iminds.iot.dianne.api.nn.Dianne;
import be.iminds.iot.dianne.api.nn.NeuralNetwork;
import be.iminds.iot.dianne.api.nn.module.AbstractModule;
import be.iminds.iot.dianne.api.nn.module.Module;
import be.iminds.iot.dianne.api.nn.module.dto.ModuleDTO;
import be.iminds.iot.dianne.api.nn.module.dto.ModuleInstanceDTO;
import be.iminds.iot.dianne.api.nn.module.dto.ModulePropertyDTO;
import be.iminds.iot.dianne.api.nn.module.dto.ModuleTypeDTO;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkDTO;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkInstanceDTO;
import be.iminds.iot.dianne.api.nn.module.factory.ModuleFactory;
import be.iminds.iot.dianne.api.nn.runtime.DianneRuntime;
import be.iminds.iot.dianne.api.repository.DianneRepository;
import be.iminds.iot.dianne.nn.util.DianneJSONConverter;
import be.iminds.iot.dianne.tensor.Tensor;

@Component(property={"aiolos.proxy=false"})
public class CompositeModuleFactory implements ModuleFactory {

	private DianneRepository repository;

	private DianneRuntime runtime;
	
	private Dianne dianne;
	
	private final Map<String, CompositeType> supportedModules = new HashMap<String, CompositeType>();
	
	private class CompositeType {
		
		public final ModuleTypeDTO type;
		public final String url;
		public final long providingBundle;
		
		public CompositeType(ModuleTypeDTO type, String url, long providingBundle){
			this.type = type;
			this.url = url;
			this.providingBundle = providingBundle;
		}
	}
	
	@Activate
	void activate(BundleContext context){
		context.addBundleListener(new BundleListener() {
			@Override
			public void bundleChanged(BundleEvent event) {
				long bundleId = event.getBundle().getBundleId();
				if(event.getType() == BundleEvent.STOPPING
					|| event.getType() == BundleEvent.UPDATED){
					Iterator<Entry<String, CompositeType>> it = supportedModules.entrySet().iterator();
					while(it.hasNext()){
						if(it.next().getValue().providingBundle==bundleId){
							it.remove();
						}
					}	
				}
				if(event.getType() == BundleEvent.STARTED
					|| event.getType() == BundleEvent.UPDATED){
					searchCompositeTypes(event.getBundle());
				}
			}
		});
		for(Bundle b : context.getBundles()){
			try {
			searchCompositeTypes(b);
			} catch(Throwable t){
				t.printStackTrace();
			}
		}
	}

	private void searchCompositeTypes(Bundle bundle){
		Enumeration<String> paths = bundle.getEntryPaths("composites");
		if(paths == null)
			return;
		
		while(paths.hasMoreElements()){
			String path = paths.nextElement();
			
			int s = path.substring(0, path.length()-1).lastIndexOf("/")+1;
			String name = path.substring(s, path.length()-1);
			String category = "Composite";
			
			// a composite module parameters is identified by a composite.txt configuration file
			// these are formatted one per line, with on each line
			// <property name>,<property key>,<property class type, i.e. java.lang.Integer>
			// these properties can be referred to in the Neural Network description as ${<property key>} 
			URL url = bundle.getEntry(path+"composite.txt");
			List<ModulePropertyDTO> props = new ArrayList<>();
			if(url != null){
				try {
				BufferedReader reader = new BufferedReader(new InputStreamReader(url.openStream()));
				String line;
				
				while((line = reader.readLine()) != null){
					String[] entries = line.split(",");
					ModulePropertyDTO prop = new ModulePropertyDTO(entries[0], entries[1], entries[2]);
					props.add(prop);
				}
				} catch(Exception e){
					System.err.println("Error parsing "+path+" composite configuration: "+e.getMessage());
				}
			}
			ModulePropertyDTO[] array = null;
			if(props.size() > 0){
				array = new ModulePropertyDTO[props.size()];
				props.toArray(array);
			}
			
			ModuleTypeDTO compositeType = new ModuleTypeDTO(name, category, true, array);
			supportedModules.put(name, new CompositeType(compositeType, bundle.getEntry("composites/"+name+"/modules.txt").toString(), bundle.getBundleId()));
		}
	}
	
	@Override
	public Module createModule(ModuleDTO dto)
			throws InstantiationException {
		return createModule(dto, null);
	}
	
	@Override
	public Module createModule(ModuleDTO dto, Tensor parameters)
			throws InstantiationException {

		AbstractModule module = null;
		
		String nnName = dto.type;
		UUID compositeId = dto.id;

		NeuralNetworkDTO nnDescription = null;
		if(supportedModules.containsKey(nnName)){
			CompositeType composite = supportedModules.get(nnName);
			// read from composites folder
			try {
				URL modules = new URL(composite.url);
				nnDescription = DianneJSONConverter.parseJSON(modules.openStream());
			} catch(Exception e){
				throw new InstantiationException("Failed to instantiate module "+dto.type+": "+e.getMessage());
			}
			
			// find any composite properties defined as ${property key} and replace
			ModuleTypeDTO compositeType = composite.type;
			for(ModuleDTO m : nnDescription.modules.values()){
				m.properties.replaceAll((key, value) -> {
					if(!value.contains("$")){
						return value;
					}
					for(ModulePropertyDTO p : compositeType.properties){
						value = value.replace("${"+p.id+"}", dto.properties.get(p.id));
					}
					if(value.contains("$")){
						int i1 = value.indexOf("$");
						int i2 = value.indexOf("}", i1);		
						throw new RuntimeException("Could not find value for "+value);
					}
					
					// figure out whether result should be an int or float
					String clazz = null;
					ModuleTypeDTO type = runtime.getSupportedModules().stream().filter(s -> s.type.equals(m.type)).findFirst().get();
					for(ModulePropertyDTO p : type.properties){
						if(p.id.equals(key)){
							clazz = p.clazz;
						}
					}
					
					value = eval(value, clazz);
					
					return value;
				});
			}
		} else {
			// try repository - in case we want to create ensembles
			try {
				nnDescription = repository.loadNeuralNetwork(nnName);
			} catch(Exception e){
				throw new InstantiationException();
			}
		}


		
		// calculate parameters Tensor size
		// TODO should this somehow be made available by the module(dto) or something?
		// This is now replicated properties parsing code from the other factory
		// and size calculation from the module impl :-(
		
		// for composite we combine training parameters and memory state ... becomes messy?
		int total = 0;
		int memory = 0;
		int size = 0;
		LinkedHashMap<UUID, Integer> parameterMapping = new LinkedHashMap<>();
		LinkedHashMap<UUID, Integer> memoryMapping = new LinkedHashMap<>();
		for(ModuleDTO m : nnDescription.modules.values()){
			switch(m.type){
			case "Linear":
				int inSize = Integer.parseInt(m.properties.get("input"));
				int outSize = Integer.parseInt(m.properties.get("output"));
				size = outSize*(inSize+1);
				parameterMapping.put(m.id, size);
				break;
			case "Convolution":
			case "FullConvolution":
				int noInputPlanes = Integer.parseInt(m.properties.get("noInputPlanes"));
				int noOutputPlanes = Integer.parseInt(m.properties.get("noOutputPlanes"));
				int kernelWidth = Integer.parseInt(m.properties.get("kernelWidth"));
				int kernelHeight = hasProperty(m.properties, "kernelHeight") ? Integer.parseInt(m.properties.get("kernelHeight")) : 1;
				int kernelDepth = hasProperty(m.properties, "kernelDepth") ? Integer.parseInt(m.properties.get("kernelDepth")) : 1;

				size = noOutputPlanes*noInputPlanes*kernelWidth*kernelHeight*kernelDepth+noOutputPlanes;
				parameterMapping.put(m.id, size);
				break;
			case "PReLU":
				size = 1;
				parameterMapping.put(m.id, size);
				break;
			case "Memory":
				size =  Integer.parseInt(m.properties.get("size"));
				memoryMapping.put(m.id, size);
				memory+=size;
				break;
			default : 
				size = 0;
				break;
			}
			total+=size;
		}
		
		// narrow for each trainable and memory module
		boolean hasParameters = true;
		if(parameters == null){
			parameters = new Tensor(total);
			parameters.fill(0.0f);
			hasParameters = false;
		} else {
			parameters.reshape(total);
		}
		
		Map<UUID, Tensor> narrowed = new HashMap<>();
		int offset = 0;
		Iterator<Entry<UUID, Integer>> it = parameterMapping.entrySet().iterator();
		while(it.hasNext()){
			Entry<UUID, Integer> e = it.next();
			int s = e.getValue();
			Tensor narrow = parameters.narrow(0, offset, s);
			narrowed.put(e.getKey(), narrow);
			
			// no parameters for composite found, load parameters for individual modules
			// useful for constructing ensembles
			if(!hasParameters){
				// try first with composite module id as tag
				try {
					Tensor t = repository.loadParameters(e.getKey(), compositeId.toString());
					t.copyInto(narrow);
				} catch (Exception e1) {
					// try again without tag
					try {
						Tensor t = repository.loadParameters(e.getKey());
						t.copyInto(narrow);
					} catch (Exception e2) {
						// no parameters found
					}
				}
			}
			
			offset+=s;
		}
		
		Iterator<Entry<UUID, Integer>> it2 = memoryMapping.entrySet().iterator();
		while(it2.hasNext()){
			Entry<UUID, Integer> e = it2.next();
			int s = e.getValue();
			Tensor narrow = parameters.narrow(0, offset, s);
			narrowed.put(e.getKey(), narrow);
			
			// TODO could one provide initial memory values? for now set to zero
			narrow.fill(0.0f);
			
			offset+=s;
		}
		
		// deploy each module and inject narrowed part of parameters
		UUID compositeNNid = UUID.randomUUID();
		
		Map<UUID, ModuleInstanceDTO> deployed = new HashMap<>();
		for(ModuleDTO m : nnDescription.modules.values()){
			try {
				ModuleInstanceDTO mi = runtime.deployModule(m, compositeNNid, narrowed.get(m.id));
				deployed.put(mi.moduleId, mi);
			} catch(Exception e){
				for(ModuleInstanceDTO mi : deployed.values()){
					runtime.undeployModule(mi);
				}
				throw new RuntimeException("Failed to deploy composite module "+compositeNNid+": "+e.getMessage(), e);
			}
		}
		NeuralNetworkInstanceDTO nnDTO = new NeuralNetworkInstanceDTO(compositeNNid, nnName, deployed);
		
		// get NeuralNetwork object
		try {
			NeuralNetwork nn = dianne.getNeuralNetwork(nnDTO).getValue();
			
			// create CompositeModule with parameters and NeuralNetwork
			Tensor params = total-memory == 0 ? null : parameters.narrow(0, total-memory);
			Tensor mem = memory == 0 ? null : parameters.narrow(total-memory, memory); 
			module = new CompositeModule(compositeId, params, mem, nn, parameterMapping);
			
		} catch (Exception e) {
			throw new RuntimeException("Failed to deploy composite module "+compositeId+": "+e.getMessage(), e);
		} 

		return module;
	}

	@Override
	public List<ModuleTypeDTO> getAvailableModuleTypes() {
		List<ModuleTypeDTO> moduleTypes = supportedModules.values().stream().map(m -> m.type).collect(Collectors.toList());
		
		// TODO do we want entries for each NN in builder toolbox?
		//for(String nnName : repository.availableNeuralNetworks()){
		//	ModuleTypeDTO type = new ModuleTypeDTO(nnName, "Networks", true);
		//	moduleTypes.add(type);
		//}
		
		return moduleTypes;
	}

	@Override
	public ModuleTypeDTO getModuleType(String name) {
		if(!supportedModules.containsKey(name)){
			return null;
		}
		return supportedModules.get(name).type;
	}
	
	// evaluate simple expressions a+b, a-b, a*b or a/c
	// for now only supports float or int return type
	private String eval(String expression, String clazz){
		int op = -1;
		op = expression.indexOf('+');
		if(op==-1)
			op = expression.indexOf('-');
		if(op==-1)
			op = expression.indexOf('*');
		if(op==-1)
			op = expression.indexOf('/');
		if(op==-1)
			return expression;
		
		
		String s1 = expression.substring(0, op);
		float a1 = Float.parseFloat(s1);
		String s2 = expression.substring(op+1);
		float a2 = Float.parseFloat(s2);
		char operator = expression.charAt(op);
		
		float result = 0;
		switch(operator){
		case '+':
			result = a1+a2;
			break;
		case '-':
			result = a1-a2;
			break;
		case '*':
			result = a1*a2;
			break;
		case '/':
			result = a1/a2;
			break;
		}
		
		if(clazz.equals("java.lang.Integer")){
			return ""+((int)result);
		} else {
			return ""+result;
		}
	}
	
	private boolean hasProperty(Map<String, String> config, String property){
		String value = (String) config.get(property);
		if(value==null){
			return false;
		} else if(value.isEmpty()){
			return false;
		}
		return true;
	}
	
	@Reference
	void setDianneRepository(DianneRepository r){
		this.repository = r;
	}
	
	@Reference
	void setDianneRuntime(DianneRuntime r){
		this.runtime = r;
	}
	
	@Reference
	void setDianne(Dianne d){
		this.dianne = d;
	}
}
