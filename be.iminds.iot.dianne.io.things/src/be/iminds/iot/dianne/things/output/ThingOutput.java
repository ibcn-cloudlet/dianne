package be.iminds.iot.dianne.things.output;

import java.util.Dictionary;
import java.util.HashMap;
import java.util.Hashtable;
import java.util.Map;
import java.util.UUID;

import org.osgi.framework.BundleContext;
import org.osgi.framework.ServiceRegistration;

import be.iminds.iot.dianne.api.io.OutputDescription;
import be.iminds.iot.dianne.api.nn.module.ForwardListener;

public abstract class ThingOutput implements ForwardListener {

	protected final UUID id;
	protected final String name;
	protected final String type;
	
	private Map<String, ServiceRegistration> registrations = new HashMap<>();
	
	public ThingOutput(UUID id, String name, String type){
		this.id = id;
		this.name = name;
		this.type = type;
	}
	
	public OutputDescription getOutputDescription(){
		return new OutputDescription(name, type);
	}
	
	public void connect(UUID nnId, UUID outputId, BundleContext context){
		String target = nnId.toString()+":"+outputId.toString();
		Dictionary<String, Object> properties = new Hashtable<String, Object>();
		properties.put("targets", new String[]{target});
		properties.put("aiolos.unique", true);
		ServiceRegistration registration = context.registerService(ForwardListener.class.getName(), this, properties);
		registrations.put(target, registration);
	}
	
	public void disconnect(UUID nnId, UUID outputId){
		String target = nnId.toString()+":"+outputId.toString();
		ServiceRegistration registration = registrations.remove(target);
		if(registration != null){
			registration.unregister();
		}
	}
	
	public void disconnect(){
		for(ServiceRegistration r : registrations.values()){
			r.unregister();
		}
		registrations.clear();
	}
	
	protected boolean isConnected(){
		return registrations.size() > 0;
	}
}
