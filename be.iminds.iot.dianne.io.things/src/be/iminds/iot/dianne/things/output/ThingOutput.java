package be.iminds.iot.dianne.things.output;

import java.util.Dictionary;
import java.util.Hashtable;
import java.util.UUID;

import org.osgi.framework.BundleContext;
import org.osgi.framework.ServiceRegistration;

import be.iminds.iot.dianne.api.io.OutputDescription;
import be.iminds.iot.dianne.api.nn.module.ForwardListener;

public abstract class ThingOutput implements ForwardListener {

	public final UUID id;
	public final String name;
	public final String type;
	
	private ServiceRegistration registration;
	
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
		registration = context.registerService(ForwardListener.class.getName(), this, properties);
	}
	
	public void disconnect(){
		if(registration != null){
			registration.unregister();
			registration = null;
		}
	}
	
}
