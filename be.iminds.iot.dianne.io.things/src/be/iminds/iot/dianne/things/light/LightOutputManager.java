package be.iminds.iot.dianne.things.light;

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

import be.iminds.iot.dianne.api.io.OutputManager;
import be.iminds.iot.dianne.api.nn.module.ForwardListener;
import be.iminds.iot.dianne.tensor.TensorFactory;
import be.iminds.iot.thing.Thing;
import be.iminds.iot.thing.light.Light;


@Component(immediate=true)
public class LightOutputManager implements OutputManager {

	private BundleContext context;
	
	private TensorFactory factory;
	
	private Map<String, Light> lights = Collections.synchronizedMap(new HashMap<String, Light>());
	
	private Map<UUID, ServiceRegistration> registrations =  Collections.synchronizedMap(new HashMap<UUID, ServiceRegistration>());
	
	@Activate
	public void activate(BundleContext context){
		this.context = context;
	}
	
	@Reference
	public void setTensorFactory(TensorFactory factory){
		this.factory = factory;
	}
	
	@Reference(
			cardinality=ReferenceCardinality.MULTIPLE, 
			policy=ReferencePolicy.DYNAMIC)
	public void addLight(Light l, Map<String, Object> properties){
		UUID id = UUID.fromString((String)properties.get(Thing.ID));
		String service = (String) properties.get(Thing.SERVICE);
		// TODO only handle local thing services?
		// TODO use id-service combo?
		lights.put(service, l);
	}
	
	public void removeLight(Light l, Map<String, Object> properties){
		UUID id = UUID.fromString((String)properties.get(Thing.ID));
		String service = (String) properties.get(Thing.SERVICE);
		lights.remove(service);
	}

	@Override
	public List<String> getAvailableOutputs() {
		ArrayList<String> outputs = new ArrayList<String>();
		synchronized(lights){
			outputs.addAll(lights.keySet());
		}
		return outputs;
	}

	@Override
	public void setOutput(UUID outputId, String output) {
		Light l = lights.get(output);
		if(l!=null){
			LightOutput o = new LightOutput(factory, l);
			Dictionary<String, Object> properties = new Hashtable<String, Object>();
			properties.put("targets", new String[]{outputId.toString()});
			properties.put("aiolos.unique", true);
			ServiceRegistration r = context.registerService(ForwardListener.class.getName(), o, properties);
			// TODO only works if outputId only forwards to one output
			registrations.put(outputId, r);
		}
	}

	@Override
	public void unsetOutput(UUID outputId, String output) {
		ServiceRegistration r = registrations.remove(outputId);
		if(r!=null){
			r.unregister();
		}
	}

}
