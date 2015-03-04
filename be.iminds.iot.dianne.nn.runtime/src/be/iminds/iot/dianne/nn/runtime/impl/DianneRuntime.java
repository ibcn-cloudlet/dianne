package be.iminds.iot.dianne.nn.runtime.impl;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Dictionary;
import java.util.Enumeration;
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
import org.osgi.service.cm.ConfigurationException;
import org.osgi.service.cm.ManagedServiceFactory;
import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Deactivate;
import org.osgi.service.component.annotations.Reference;
import org.osgi.service.component.annotations.ReferenceCardinality;
import org.osgi.service.component.annotations.ReferencePolicy;

import be.iminds.iot.dianne.nn.module.Input;
import be.iminds.iot.dianne.nn.module.Module;
import be.iminds.iot.dianne.nn.module.Output;
import be.iminds.iot.dianne.nn.module.Preprocessor;
import be.iminds.iot.dianne.nn.module.Trainable;
import be.iminds.iot.dianne.nn.module.description.ModuleType;
import be.iminds.iot.dianne.nn.module.factory.ModuleFactory;
import be.iminds.iot.dianne.nn.runtime.ModuleManager;
import be.iminds.iot.dianne.repository.DianneRepository;
import be.iminds.iot.dianne.tensor.TensorFactory;

@Component(immediate=true, 
	property={"service.pid=be.iminds.iot.dianne.nn.module",
			  "aiolos.callback=be.iminds.iot.dianne.nn.runtime.ModuleManager"})
public class DianneRuntime implements ManagedServiceFactory, ModuleManager {

	private BundleContext context;
	
	// TODO support multiple factories in the future?!
	private TensorFactory tFactory;
	private List<ModuleFactory> mFactories = Collections.synchronizedList(new ArrayList<ModuleFactory>());
	
	private DianneRepository repository;
	
	// All module UUIDs by their PID
	private Map<String, UUID> uuids = new HashMap<String, UUID>();
	// All known modules by their UUID
	private Map<UUID, Module> modules = new HashMap<UUID, Module>();
	// All module service registrations by their UUID
	private Map<UUID, ServiceRegistration> registrations = new HashMap<UUID, ServiceRegistration>();
	
	private Map<UUID, List<UUID>> nextMap = new HashMap<UUID, List<UUID>>();
	private Map<UUID, List<UUID>> prevMap = new HashMap<UUID, List<UUID>>();
	
	@Override
	public String getName() {
		return "be.iminds.iot.dianne.nn.module";
	}

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
	
	@Reference
	public void setTensorFactory(TensorFactory factory){
		this.tFactory = factory;
	}
	
	@Reference(cardinality=ReferenceCardinality.AT_LEAST_ONE, 
			policy=ReferencePolicy.DYNAMIC)
	public void addModuleFactory(ModuleFactory factory){
		this.mFactories.add(factory);
	}
	
	public void removeModuleFactory(ModuleFactory factory){
		this.mFactories.remove(factory);
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
	
	
	@Override
	public synchronized void updated(String pid, Dictionary<String, ?> properties)
			throws ConfigurationException {
		try {
			UUID id = deployModule(properties);
			this.uuids.put(pid, id);
		} catch(InstantiationException e){
			System.err.println("Failed to instantiate module defined by "+pid);
		}
	}

	@Override
	public synchronized void deleted(String pid) {
		UUID id = uuids.remove(pid);
		undeployModule(id);
	}
	
	@Override
	public synchronized UUID deployModule(Dictionary<String, ?> properties) throws InstantiationException{
		// Create and register module
		Module module = null;
		synchronized(mFactories){
			Iterator<ModuleFactory> it = mFactories.iterator();
			while(module == null && it.hasNext()){
				try {
					ModuleFactory mFactory = it.next();
					module = mFactory.createModule(tFactory, properties);
				} catch(InstantiationException e){
					
				}
			}
		}

		if(module==null){
			throw new InstantiationException("Failed to instantiate module");
		}
		
		// configure next/prev
		String next = (String)properties.get("module.next");
		List<UUID> nextIDs =  parseUUIDs(next);
		nextMap.put(module.getId(), nextIDs);
		configureNext(module);
		
		String prev = (String)properties.get("module.prev");
		List<UUID> prevIDs = parseUUIDs(prev);
		prevMap.put(module.getId(), prevIDs);
		configurePrevious(module);

		// set labels in case of output
		if(module instanceof Output){
			String labels = (String)properties.get("module.labels");
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
		for(Enumeration<String> keys = properties.keys();keys.hasMoreElements();){
			String key = keys.nextElement();
			props.put(key, properties.get(key));
		}
		// make sure that for each module all interfaces are behind a single proxy 
		// and that each module is uniquely proxied
		props.put("aiolos.combine", "*");
		props.put("aiolos.instance.id", module.getId().toString());
		
		// register on behalf of bundle that provided configuration if applicable
		BundleContext c = context;
		Long bundleId = (Long) properties.get("be.iminds.aiolos.configurer.bundle.id");
		if(bundleId!=null){
			c = context.getBundle(bundleId).getBundleContext();
		}
		
		ServiceRegistration reg = c.registerService(classes, module, props);
		this.registrations.put(module.getId(), reg);
		
		
		System.out.println("Registered module "+module.getClass().getName()+" "+module.getId());
		return null;
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
	public List<ModuleType> getSupportedModules() {
		List<ModuleType> supported = new ArrayList<ModuleType>();
		synchronized(mFactories){
			for(ModuleFactory f : mFactories){
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
