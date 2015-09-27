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

import be.iminds.iot.dianne.api.nn.Dianne;
import be.iminds.iot.dianne.api.nn.NeuralNetwork;
import be.iminds.iot.dianne.api.nn.module.Module;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkInstanceDTO;
import be.iminds.iot.dianne.api.repository.DianneRepository;
import be.iminds.iot.dianne.tensor.TensorFactory;

@Component(property={"aiolos.export=false"})
public class DianneImpl implements Dianne {

	private BundleContext context;
	
	private TensorFactory factory;
	private DianneRepository repository;
	
	private Map<UUID, Map<UUID, Module>> modules = Collections.synchronizedMap(new HashMap<>());
	
	private Map<UUID, NeuralNetworkWrapper> nns = Collections.synchronizedMap(new HashMap<UUID, NeuralNetworkWrapper>());
	private Map<UUID, NeuralNetwork> nnServices = Collections.synchronizedMap(new HashMap<UUID, NeuralNetwork>());
	
	
	@Override
	public NeuralNetwork getNeuralNetwork(NeuralNetworkInstanceDTO nni) {
		NeuralNetwork nn = nnServices.get(nni.id);
		if(nn!=null){
			return nn;
		}
		
		// try to create new NeuralNetworkWrapper for this instance
		Map<UUID, Module> m = modules.get(nni.id);
		if(m.size() == nni.modules.size()){
			// all modules available
			NeuralNetworkWrapper wrapper = new NeuralNetworkWrapper(nni, m.values(), repository, factory, context);
			wrapper.register();
			nns.put(nni.id, wrapper);
		}
		
		return nnServices.get(nni.id);
	}

	@Activate
	public void activate(BundleContext ctx){
		this.context = ctx;
	}
	
	@Reference
	void setDianneRepository(DianneRepository r){
		repository = r;
	}
	
	@Reference
	void setTensorFactory(TensorFactory f){
		factory = f;
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
		nn.unregister();
	}
	
	@Reference(cardinality=ReferenceCardinality.MULTIPLE, 
			policy=ReferencePolicy.DYNAMIC)
	public void addNeuralNetwork(NeuralNetwork nn, Map<String, Object> properties){
		String nnId = (String)properties.get("nn.id");
		nnServices.put(UUID.fromString(nnId), nn);
	}
	
	public void removeNeuralNetwork(NeuralNetwork nn, Map<String, Object> properties){
		String nnId = (String)properties.get("nn.id");
		nnServices.remove(UUID.fromString(nnId));
	}
	
}
