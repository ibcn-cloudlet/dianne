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
import org.osgi.framework.Constants;
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
import be.iminds.iot.dianne.api.nn.module.dto.ModuleInstanceDTO;
import be.iminds.iot.dianne.api.nn.module.dto.ModuleTypeDTO;
import be.iminds.iot.dianne.api.nn.module.factory.ModuleFactory;
import be.iminds.iot.dianne.api.nn.runtime.ModuleManager;
import be.iminds.iot.dianne.api.repository.DianneRepository;
import be.iminds.iot.dianne.tensor.Tensor;

@Component(immediate=true, 
	property={"service.pid=be.iminds.iot.dianne.nn.module",
			  "aiolos.callback=be.iminds.iot.dianne.api.nn.runtime.ModuleManager"})
public class DianneRuntime implements ModuleManager {

	private BundleContext context;
	private UUID runtimeId;
	
	private List<ModuleFactory> moduleFactories = Collections.synchronizedList(new ArrayList<ModuleFactory>());
	
	private DianneRepository repository;
	
	// All known modules
	private ModuleMap<Module> modules = new ModuleMap<Module>();
	// All module service registrations 
	private ModuleMap<ServiceRegistration> registrations = new ModuleMap<ServiceRegistration>();
	private ModuleMap<ModuleInstanceDTO> instances = new ModuleMap<ModuleInstanceDTO>();
	
	private Map<UUID, List<UUID>> nextMap = new HashMap<UUID, List<UUID>>();
	private Map<UUID, List<UUID>> prevMap = new HashMap<UUID, List<UUID>>();
	
	// Listener targets are defined as nnId:moduleId
	private Map<ForwardListener, List<String>> forwardListeners = new HashMap<ForwardListener, List<String>>();
	private Map<BackwardListener, List<String>> backwardListeners = new HashMap<BackwardListener, List<String>>();

	@Activate
	public void activate(BundleContext context){
		this.context = context;
		this.runtimeId = UUID.fromString(context.getProperty(Constants.FRAMEWORK_UUID));
	}
	
	@Deactivate
	public void deactivate(){
		synchronized(registrations){
			for(ServiceRegistration reg : registrations.values()){
				reg.unregister();
			}
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
		UUID moduleId = UUID.fromString((String)properties.get("module.id"));
		UUID nnId = UUID.fromString((String)properties.get("nn.id"));

		modules.put(moduleId, nnId, module);
		
		// configure modules that require this module
		for(Module m : findDependingModules(moduleId, nnId, nextMap)){
			configureNext(m, nnId);
		}
		for(Module m : findDependingModules(moduleId, nnId, prevMap)){
			configurePrevious(m, nnId);
		}
	
		configureModuleListeners(moduleId, nnId, module);
	}
	
	public synchronized void updatedModule(Module module, Map<String, Object> properties){
		UUID moduleId = UUID.fromString((String)properties.get("module.id"));
		UUID nnId = UUID.fromString((String)properties.get("nn.id"));
		configureModuleListeners(moduleId, nnId, module);
	}
	
	private void configureModuleListeners(UUID moduleId, UUID nnId, Module module){
		if(registrations.containsKey(moduleId, nnId)){
			// check if someone is listening for this (locally registered) module
			synchronized(forwardListeners){
				Iterator<Entry<ForwardListener, List<String>>> it = forwardListeners.entrySet().iterator();
				while(it.hasNext()){
					Entry<ForwardListener, List<String>> e = it.next();
					for(String target : e.getValue()){
						configureForwardListener(e.getKey(), moduleId, nnId, module, target);
					}
				}
			}
			
			synchronized(backwardListeners){
				Iterator<Entry<BackwardListener, List<String>>> it = backwardListeners.entrySet().iterator();
				while(it.hasNext()){
					Entry<BackwardListener, List<String>> e = it.next();
					for(String target : e.getValue()){
						configureBackwardListener(e.getKey(), moduleId, nnId, module, target);
					}
				}
			}
		}
	}
	
	private void configureForwardListener(ForwardListener l, UUID moduleId, UUID nnId, Module module, String target){
		String[] split = target.split(":");
		if(split.length==1){
			if(target.contains(":")){
				// only moduleId
				UUID mid = UUID.fromString(split[0]);
				if(mid.equals(moduleId)){
					module.addForwardListener(l);
				}
			} else {
				// only nnId
				UUID nid = UUID.fromString(split[0]);
				if(nid.equals(nnId)){
					// only add to output modules 
					if(module instanceof Output){
						module.addForwardListener(l);
					}
				}
			}
		} else {
			// nnId:moduleId
			UUID nid = UUID.fromString(split[0]);
			UUID mid = UUID.fromString(split[1]);
			if(nid.equals(nnId) && mid.equals(moduleId)){
				module.addForwardListener(l);
			}
		}
	}
	
	private void configureBackwardListener(BackwardListener l, UUID moduleId, UUID nnId, Module module, String target){
		String[] split = target.split(":");
		if(split.length==1){
			if(target.contains(":")){
				// only moduleId
				UUID mid = UUID.fromString(split[0]);
				if(mid.equals(moduleId)){
					module.addBackwardListener(l);
				}
			} else {
				// only nnId
				UUID nid = UUID.fromString(split[0]);
				if(nid.equals(nnId)){
					// only add to input modules 
					if(module instanceof Input){
						module.addBackwardListener(l);
					}
				}
			}
		} else {
			// nnId:moduleId
			UUID nid = UUID.fromString(split[0]);
			UUID mid = UUID.fromString(split[1]);
			if(nid.equals(nnId) && mid.equals(moduleId)){
				module.addBackwardListener(l);
			}
		}
	}
	
	public synchronized void removeModule(Module module, Map<String, Object> properties){
		UUID moduleId = UUID.fromString((String)properties.get("module.id"));
		UUID nnId = UUID.fromString((String)properties.get("nn.id"));
		modules.remove(moduleId, nnId);
		
		// unconfigure modules that require this module
		for(Module m : findDependingModules(moduleId, nnId, nextMap)){
			unconfigureNext(m);
		}
		for(Module m : findDependingModules(moduleId, nnId, prevMap)){
			unconfigurePrevious(m);
		}
	}
	
	@Reference(
			cardinality=ReferenceCardinality.MULTIPLE, 
			policy=ReferencePolicy.DYNAMIC)
	public synchronized void addForwardListener(ForwardListener l, Map<String, Object> properties){
		String[] targets = (String[])properties.get("targets");
		if(targets!=null){
			for(String target : targets){
				// TODO filter out the modules that should match this target 
				// instead of iterating all and trying?
				synchronized(modules){
					Iterator<ModuleMap<Module>.Entry<Module>> it = modules.iterator();
					while(it.hasNext()){
						ModuleMap<Module>.Entry<Module> e = it.next();
						configureForwardListener(l, e.moduleId, e.nnId, e.value, target);
					}
				}
			}
			forwardListeners.put(l, Arrays.asList(targets));
		}
	}
	
	public synchronized void removeForwardListener(ForwardListener l){
		List<String> targets = forwardListeners.remove(l);
		// TODO filter out the modules that actually have this listener registered?
		synchronized(modules){
			for(Module m : modules.values()){
				m.removeForwardListener(l);
			}
		}
	}
	
	
	@Reference(
			cardinality=ReferenceCardinality.MULTIPLE, 
			policy=ReferencePolicy.DYNAMIC)
	public synchronized void addBackwardListener(BackwardListener l, Map<String, Object> properties){
		String[] targets = (String[])properties.get("targets");
		if(targets!=null){
			for(String target : targets){
				// TODO filter out the modules that should match this target 
				// instead of iterating all and trying?
				synchronized(modules){
					Iterator<ModuleMap<Module>.Entry<Module>> it = modules.iterator();
					while(it.hasNext()){
						ModuleMap<Module>.Entry<Module> e = it.next();
						configureBackwardListener(l, e.moduleId, e.nnId, e.value, target);
					}
				}
			}
			backwardListeners.put(l, Arrays.asList(targets));
		}
	}
	
	public synchronized void removeBackwardListener(BackwardListener l){
		List<String> targets = backwardListeners.remove(l);
		// TODO filter out the modules that actually have this listener registered?
		synchronized(modules){
			for(Module m : modules.values()){
				m.removeBackwardListener(l);
			}
		}
	}
	
	@Override
	public synchronized ModuleInstanceDTO deployModule(ModuleDTO dto, UUID nnId) throws InstantiationException{
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
		configureNext(module, nnId);
		
		
		List<UUID> prevIDs = new ArrayList<>();
		if(dto.prev!=null){
			for(UUID id : dto.prev){
				prevIDs.add(id);
			}
		}
		prevMap.put(module.getId(), prevIDs);
		configurePrevious(module, nnId);

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
				Tensor parameters = repository.loadParameters(module.getId());
				((Trainable)module).setParameters(parameters);
			} catch(IOException e){}
		} else if(module instanceof Preprocessor){
			try {
				Tensor parameters = repository.loadParameters(module.getId());
				((Preprocessor)module).setParameters(parameters);
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
		
		UUID moduleId = module.getId();
		
		Dictionary<String, Object> props = new Hashtable<String, Object>();
		props.put("module.id", moduleId.toString());
		props.put("module.type", dto.type);
		props.put("nn.id", nnId.toString());
		
		// make sure that for each module all interfaces are behind a single proxy 
		// and that each module is uniquely proxied
		props.put("aiolos.combine", "*");
		props.put("aiolos.instance.id", nnId.toString()+":"+module.getId().toString());
		
		// allready add a null registration, in order to allow registrations.contains()
		// to return true in the addModule call of this class
		this.registrations.put(moduleId, nnId, null);
		ServiceRegistration reg = context.registerService(classes, module, props);
		this.registrations.put(moduleId, nnId, reg);
		
		System.out.println("Registered module "+module.getClass().getName()+" "+moduleId);
		ModuleInstanceDTO instance =  new ModuleInstanceDTO(dto, nnId, runtimeId);
		this.instances.put(moduleId, nnId, instance);
		return instance;
	}

	@Override
	public synchronized void undeployModule(ModuleInstanceDTO dto) {
		if(!dto.runtimeId.equals(runtimeId)){
			System.out.println("Can only undeploy module instances that are deployed here...");
			return;
		}
		
		ServiceRegistration reg = registrations.remove(dto.moduleId, dto.moduleId);
		if(reg!=null){
			try {
				reg.unregister();
			} catch(IllegalStateException e){
				// happens when the service was registered on behalf of the (config) bundle
				// that is uninstalled (then service is allready unregistered)
			}
		}
		
		instances.remove(dto.moduleId, dto.nnId);
		
		if(!registrations.containsKey(dto.moduleId)){
			nextMap.remove(dto.moduleId);
			prevMap.remove(dto.moduleId);
		}
		
		System.out.println("Unregistered module "+dto.moduleId);
	}
	
	@Override
	public synchronized void undeployModules(UUID nnId) {
		List<ModuleInstanceDTO> toRemove = new ArrayList<ModuleInstanceDTO>();
		synchronized(instances){
			Iterator<ModuleMap<ModuleInstanceDTO>.Entry<ModuleInstanceDTO>> it = instances.iterator();
			while(it.hasNext()){
				ModuleMap<ModuleInstanceDTO>.Entry<ModuleInstanceDTO> e = it.next();
				if(e.nnId.equals(nnId)){
					toRemove.add(e.value);
				}
			}
		}
		
		for(ModuleInstanceDTO m : toRemove){
			undeployModule(m);
		}
	}
	
	@Override
	public List<ModuleInstanceDTO> getModules(){
		List<ModuleInstanceDTO> modules = new ArrayList<ModuleInstanceDTO>();
		modules.addAll(instances.values());
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
	
	@Override
	public UUID getRuntimeId() {
		return runtimeId;
	}
	
	private void configureNext(Module m, UUID nnId){
		List<UUID> nextIDs = nextMap.get(m.getId());
		if(nextIDs.size()==0){
			// output module
			return;
		}
		Module[] nextModules = new Module[nextIDs.size()];
		
		int i = 0;
		for(UUID nextID : nextIDs){
			Module nextModule = modules.get(nextID, nnId);
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
	
	private void configurePrevious(Module m, UUID nnId){
		List<UUID> prevIDs = prevMap.get(m.getId());
		if(prevIDs.size()==0){
			// input module
			return;
		}
		Module[] prevModules = new Module[prevIDs.size()];
		
		int i = 0;
		for(UUID prevID : prevIDs){
			Module prevModule = modules.get(prevID, nnId);
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
	
	private List<Module> findDependingModules(UUID moduleId, UUID nnId, Map<UUID, List<UUID>> map){
		List<Module> result = new ArrayList<Module>();
		for(Iterator<Entry<UUID, List<UUID>>> it = map.entrySet().iterator();it.hasNext();){
			Entry<UUID, List<UUID>> entry = it.next();
			for(UUID nxtId : entry.getValue()){
				if(nxtId.equals(moduleId)){
					Module m = modules.get(entry.getKey(), nnId);
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
