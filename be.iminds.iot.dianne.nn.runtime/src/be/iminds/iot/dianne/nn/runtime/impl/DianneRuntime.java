package be.iminds.iot.dianne.nn.runtime.impl;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Dictionary;
import java.util.HashMap;
import java.util.Hashtable;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.StringTokenizer;
import java.util.UUID;

import org.osgi.framework.BundleContext;
import org.osgi.framework.ServiceRegistration;
import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Deactivate;
import org.osgi.service.component.annotations.Reference;
import org.osgi.service.component.annotations.ReferenceCardinality;
import org.osgi.service.component.annotations.ReferencePolicy;

import be.iminds.iot.dianne.api.nn.module.BackwardListener;
import be.iminds.iot.dianne.api.nn.module.ForwardListener;
import be.iminds.iot.dianne.api.nn.module.Input;
import be.iminds.iot.dianne.api.nn.module.Module;
import be.iminds.iot.dianne.api.nn.module.Output;
import be.iminds.iot.dianne.api.nn.module.Preprocessor;
import be.iminds.iot.dianne.api.nn.module.Trainable;
import be.iminds.iot.dianne.api.nn.module.dto.ModuleDTO;
import be.iminds.iot.dianne.api.nn.module.dto.ModuleTypeDTO;
import be.iminds.iot.dianne.api.nn.module.factory.ModuleFactory;
import be.iminds.iot.dianne.api.nn.runtime.ModuleManager;
import be.iminds.iot.dianne.api.repository.DianneRepository;

@Component(immediate=true, 
	property={"service.pid=be.iminds.iot.dianne.nn.module",
			  "aiolos.callback=be.iminds.iot.dianne.api.nn.runtime.ModuleManager"})
public class DianneRuntime implements ModuleManager {

	private BundleContext context;
	
	private List<ModuleFactory> moduleFactories = Collections.synchronizedList(new ArrayList<ModuleFactory>());
	
	private DianneRepository repository;
	
	// All module UUIDs by their PID
	private Map<String, UUID> uuids = new HashMap<String, UUID>();
	// All known modules by their UUID
	private Map<UUID, Module> modules = new HashMap<UUID, Module>();
	// All module service registrations by their UUID
	private Map<UUID, ServiceRegistration> registrations = Collections.synchronizedMap(new HashMap<UUID, ServiceRegistration>());
	
	private Map<UUID, List<UUID>> nextMap = new HashMap<UUID, List<UUID>>();
	private Map<UUID, List<UUID>> prevMap = new HashMap<UUID, List<UUID>>();
	
	private Map<ForwardListener, List<UUID>> forwardListeners = new HashMap<ForwardListener, List<UUID>>();
	private Map<BackwardListener, List<UUID>> backwardListeners = new HashMap<BackwardListener, List<UUID>>();

	@Activate
	public void activate(BundleContext context){
		this.context = context;
	}
	
	@Deactivate
	public void deactivate(){
		for(ServiceRegistration reg : registrations.values()){
			reg.unregister();
		}
	}
	
	@Reference(cardinality=ReferenceCardinality.AT_LEAST_ONE, 
			policy=ReferencePolicy.DYNAMIC)
	public void addModuleFactory(ModuleFactory factory){
		this.moduleFactories.add(factory);
	}
	
	public void removeModuleFactory(ModuleFactory factory){
		this.moduleFactories.remove(factory);
	}
	
	@Reference
	public void setDianneRepository(DianneRepository repo){
		this.repository = repo;
	}
	
	@Reference(
			cardinality=ReferenceCardinality.MULTIPLE, 
			policy=ReferencePolicy.DYNAMIC)
	public synchronized void addModule(Module module, Map<String, Object> properties){
		UUID id = UUID.fromString((String)properties.get("module.id"));
		modules.put(id, module);
		
		// configure modules that require this module
		for(Module m : findDependingModules(id, nextMap)){
			configureNext(m);
		}
		for(Module m : findDependingModules(id, prevMap)){
			configurePrevious(m);
		}
	
		configureModuleListeners(id, module);
	}
	
	public synchronized void updatedModule(Module module, Map<String, Object> properties){
		UUID id = UUID.fromString((String)properties.get("module.id"));
		configureModuleListeners(id, module);
	}
	
	private void configureModuleListeners(UUID id, Module module){
		if(registrations.containsKey(id)){
			// check if someone is listening for this (locally registered) module
			synchronized(forwardListeners){
				Iterator<Entry<ForwardListener, List<UUID>>> it = forwardListeners.entrySet().iterator();
				while(it.hasNext()){
					Entry<ForwardListener, List<UUID>> e = it.next();
					for(UUID i : e.getValue()){
						if(id.equals(i)){
							module.addForwardListener(e.getKey());
						}
					}
				}
			}
			
			synchronized(backwardListeners){
				Iterator<Entry<BackwardListener, List<UUID>>> it = backwardListeners.entrySet().iterator();
				while(it.hasNext()){
					Entry<BackwardListener, List<UUID>> e = it.next();
					for(UUID i : e.getValue()){
						if(id.equals(i)){
							module.addBackwardListener(e.getKey());
						}
					}
				}
			}
		}
	}
	
	public synchronized void removeModule(Module module, Map<String, Object> properties){
		UUID id = UUID.fromString((String)properties.get("module.id"));
		modules.remove(id);
		
		// unconfigure modules that require this module
		for(Module m : findDependingModules(id, nextMap)){
			unconfigureNext(m);
		}
		for(Module m : findDependingModules(id, prevMap)){
			unconfigurePrevious(m);
		}
	}
	
	@Reference(
			cardinality=ReferenceCardinality.MULTIPLE, 
			policy=ReferencePolicy.DYNAMIC)
	public synchronized void addForwardListener(ForwardListener l, Map<String, Object> properties){
		String[] targets = (String[])properties.get("targets");
		if(targets!=null){
			List<UUID> ids = new ArrayList<UUID>();
			for(String t : targets){
				UUID id = UUID.fromString(t);
				ids.add(id);
				if(registrations.containsKey(id)){
					Module m = modules.get(id);
					if(m!=null){ // should not be null?
						m.addForwardListener(l);
					}
				}
			}
			forwardListeners.put(l, ids);
		}
	}
	
	public synchronized void removeForwardListener(ForwardListener l){
		List<UUID> ids = forwardListeners.remove(l);
		if(ids!=null){
			for(UUID id : ids){
				if(registrations.containsKey(id)){
					Module m = modules.get(id);
					if(m!=null){ // should not be null?
						m.removeForwardListener(l);
					}
				}
			}
		}
	}
	
	
	@Reference(
			cardinality=ReferenceCardinality.MULTIPLE, 
			policy=ReferencePolicy.DYNAMIC)
	public synchronized void addBackwardListener(BackwardListener l, Map<String, Object> properties){
		String[] targets = (String[])properties.get("targets");
		if(targets!=null){
			List<UUID> ids = new ArrayList<UUID>();
			for(String t : targets){
				UUID id = UUID.fromString(t);
				ids.add(id);
				if(registrations.containsKey(id)){
					Module m = modules.get(id);
					if(m!=null){ // should not be null?
						m.addBackwardListener(l);
					}
				}
			}
			backwardListeners.put(l, ids);
		}
	}
	
	public synchronized void removeBackwardListener(BackwardListener l){
		List<UUID> ids = backwardListeners.remove(l);
		if(ids!=null){
			for(UUID id : ids){
				if(registrations.containsKey(id)){
					Module m = modules.get(id);
					if(m!=null){ // should not be null?
						m.removeBackwardListener(l);
					}
				}
			}
		}
	}
	
	@Override
	public synchronized UUID deployModule(ModuleDTO dto) throws InstantiationException{
		// Create and register module
		Module module = null;
		synchronized(moduleFactories){
			Iterator<ModuleFactory> it = moduleFactories.iterator();
			while(module == null && it.hasNext()){
				try {
					ModuleFactory mFactory = it.next();
					module = mFactory.createModule(dto);
				} catch(InstantiationException e){
					
				}
			}
		}

		if(module==null){
			throw new InstantiationException("Failed to instantiate module");
		}
		
		// configure next/prev
		List<UUID> nextIDs = new ArrayList<>();
		if(dto.next!=null){
			for(UUID id : dto.next){
				nextIDs.add(id);
			}
		}
		nextMap.put(module.getId(), nextIDs);
		configureNext(module);
		
		
		List<UUID> prevIDs = new ArrayList<>();
		if(dto.prev!=null){
			for(UUID id : dto.prev){
				prevIDs.add(id);
			}
		}
		prevMap.put(module.getId(), prevIDs);
		configurePrevious(module);

		// set labels in case of output
		if(module instanceof Output){
			String labels = dto.properties.get("labels");
			if(labels!=null){
				String[] l = parseStrings(labels);
				((Output)module).setOutputLabels(l);
			}
		}
		
		if(module instanceof Trainable){
			try {
				float[] weights = repository.loadWeights(module.getId());
				((Trainable)module).setParameters(weights);
			} catch(IOException e){}
		} else if(module instanceof Preprocessor){
			try {
				float[] weights = repository.loadWeights(module.getId());
				((Preprocessor)module).setParameters(weights);
			} catch(IOException e){}
		} 
		
		String[] classes;
		if(module instanceof Input){
			classes = new String[]{Module.class.getName(),Input.class.getName()};
		}else if(module instanceof Output){
			classes = new String[]{Module.class.getName(),Output.class.getName()};
		} else if(module instanceof Trainable){
			classes = new String[]{Module.class.getName(),Trainable.class.getName()};
		} else if(module instanceof Preprocessor){
			classes = new String[]{Module.class.getName(),Preprocessor.class.getName()};
		} else {
			classes = new String[]{Module.class.getName()};
		}
		
		Dictionary<String, Object> props = new Hashtable<String, Object>();
		props.put("module.id", module.getId().toString());
		
		// make sure that for each module all interfaces are behind a single proxy 
		// and that each module is uniquely proxied
		props.put("aiolos.combine", "*");
		props.put("aiolos.instance.id", module.getId().toString());
		
		UUID id = module.getId();
		// allready add a null registration, in order to allow registrations.contains()
		// to return true in the addModule call of this class
		this.registrations.put(id, null);
		ServiceRegistration reg = context.registerService(classes, module, props);
		this.registrations.put(id, reg);
		
		System.out.println("Registered module "+module.getClass().getName()+" "+id);
		return id;
	}

	@Override
	public synchronized void undeployModule(UUID id) {
		nextMap.remove(id);
		prevMap.remove(id);
		ServiceRegistration reg = registrations.remove(id);
		if(reg!=null){
			try {
				reg.unregister();
			} catch(IllegalStateException e){
				// happens when the service was registered on behalf of the (config) bundle
				// that is uninstalled (then service is allready unregistered)
			}
		}
		System.out.println("Unregistered module "+id);
	}
	
	@Override
	public List<UUID> getModules(){
		List<UUID> modules = new ArrayList<UUID>();
		synchronized(registrations){
			for(UUID m : registrations.keySet()){
				modules.add(m);
			}
		}
		return modules;
	}

	@Override
	public List<ModuleTypeDTO> getSupportedModules() {
		List<ModuleTypeDTO> supported = new ArrayList<ModuleTypeDTO>();
		synchronized(moduleFactories){
			for(ModuleFactory f : moduleFactories){
				supported.addAll(f.getAvailableModuleTypes());
			}
		}
		return Collections.unmodifiableList(supported);
	}
	
	private void configureNext(Module m){
		List<UUID> nextIDs = nextMap.get(m.getId());
		if(nextIDs.size()==0){
			// output module
			return;
		}
		Module[] nextModules = new Module[nextIDs.size()];
		
		int i = 0;
		for(UUID nextID : nextIDs){
			Module nextModule = modules.get(nextID);
			// TODO also allow only partly deployed NNs?
			if(nextModule== null)
				return;
			
			nextModules[i] = nextModule;
			i++;
		}
		
		m.setNext(nextModules);
	}
	
	private void unconfigureNext(Module m){
		m.setNext((Module[]) null);
	}
	
	private void configurePrevious(Module m){
		List<UUID> prevIDs = prevMap.get(m.getId());
		if(prevIDs.size()==0){
			// input module
			return;
		}
		Module[] prevModules = new Module[prevIDs.size()];
		
		int i = 0;
		for(UUID prevID : prevIDs){
			Module prevModule = modules.get(prevID);
			if(prevModule== null)
				return;
			
			prevModules[i] = prevModule;
			i++;
		}
		
		m.setPrevious(prevModules);
	}
	
	private void unconfigurePrevious(Module m){
		m.setPrevious((Module[])null);
	}
	
	private List<Module> findDependingModules(UUID id, Map<UUID, List<UUID>> map){
		List<Module> result = new ArrayList<Module>();
		for(Iterator<Entry<UUID, List<UUID>>> it = map.entrySet().iterator();it.hasNext();){
			Entry<UUID, List<UUID>> entry = it.next();
			for(UUID nxtId : entry.getValue()){
				if(nxtId.equals(id)){
					Module m = modules.get(entry.getKey());
					if(m!=null) // could be null if removed by external bundle stop
						result.add(m);
				}
			}
		}
		return result;
	}
	
	private List<UUID> parseUUIDs(String string){
		ArrayList<UUID> result = new ArrayList<UUID>();
		if(string!=null){
			StringTokenizer st = new StringTokenizer(string, ",");
			while(st.hasMoreTokens()){
				UUID id = UUID.fromString(st.nextToken());
				result.add(id);
			}
		}
		return result;
	}
	
	private float[] parseWeights(String string){
		String[] strings = parseStrings(string);
		float weights[] = new float[strings.length];
		for (int i = 0; i < weights.length; i++) {
			weights[i] = Float.parseFloat(strings[i]);
		}
		return weights;
	}
	
	private String[] parseStrings(String string){
		String[] strings = string.replace("[", "").replace("]", "").split(", ");
		return strings;
	}
}
