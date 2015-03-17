package be.iminds.iot.dianne.things.camera;

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

import be.iminds.iot.dianne.api.nn.module.Input;
import be.iminds.iot.dianne.api.io.InputManager;
import be.iminds.iot.dianne.api.nn.module.Module;
import be.iminds.iot.dianne.tensor.TensorFactory;
import be.iminds.iot.thing.Thing;
import be.iminds.iot.thing.camera.Camera;
import be.iminds.iot.thing.camera.CameraListener;
import be.iminds.iot.thing.light.Light;

@Component(immediate=true)
public class CameraInputManager implements InputManager {

	private BundleContext context;
	
	private TensorFactory factory;
	
	private Map<String, UUID> cameraIds = Collections.synchronizedMap(new HashMap<String, UUID>());
	private Map<UUID, Camera> cameras = Collections.synchronizedMap(new HashMap<UUID, Camera>());

	private Map<UUID, Input> inputs = Collections.synchronizedMap(new HashMap<UUID, Input>());
	
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
	public void addCamera(Camera c, Map<String, Object> properties){
		UUID id = UUID.fromString((String)properties.get(Thing.ID));
		String service = (String) properties.get(Thing.SERVICE);
		
		String[] parts = service.split("_");
		String filtered = "Webcam "+parts[1];
		// TODO only handle local thing services?
		// TODO use id-service combo?
		cameraIds.put(filtered, id);
		cameras.put(id, c);
		
	}
	
	public void removeCamera(Camera c, Map<String, Object> properties){
		UUID id = UUID.fromString((String)properties.get(Thing.ID));
		String service = (String) properties.get(Thing.SERVICE);
		
		String[] parts = service.split("_");
		String filtered = parts[1];
		cameraIds.remove(filtered);
		cameras.remove(id);
		
	}
	
	@Reference(cardinality=ReferenceCardinality.MULTIPLE, 
			policy=ReferencePolicy.DYNAMIC)
	public void addInput(Input i, Map<String, Object> properties){
		String id = (String)properties.get("aiolos.instance.id");
		inputs.put(UUID.fromString(id), i);
	}
	
	public void removeInput(Input i, Map<String, Object> properties){
		String id = (String)properties.get("aiolos.instance.id");
		inputs.remove(UUID.fromString(id));
	}
	
	@Override
	public List<String> getAvailableInputs() {
		ArrayList<String> inputs = new ArrayList<String>();
		synchronized(cameraIds){
			inputs.addAll(cameraIds.keySet());
		}
		return inputs;
	}

	@Override
	public void setInput(UUID inputId, String input) {
		UUID cameraId = cameraIds.get(input);
		if(cameraId!=null){
			Camera camera = cameras.get(cameraId);
			if(camera!=null){
				camera.start(320, 240);
			}
			
			CameraInput i = new CameraInput(factory, inputs.get(inputId));
			Dictionary<String, Object> properties = new Hashtable<String, Object>();
			properties.put(CameraListener.CAMERA_ID, cameraId.toString());
			properties.put(CameraListener.CAMERA_FORMAT, Camera.Format.GRAYSCALE);
			properties.put("aiolos.unique", true);
			ServiceRegistration r = context.registerService(CameraListener.class.getName(), i, properties);
			// TODO only works if outputId only forwards to one output
			registrations.put(inputId, r);
		}
	}

	@Override
	public void unsetInput(UUID inputId, String input) {
		UUID cameraId = cameraIds.get(input);
		if(cameraId!=null){
			Camera camera = cameras.get(cameraId);
			if(camera!=null){
				camera.stop();
			}
		}
		
		ServiceRegistration r = registrations.remove(inputId);
		if(r!=null){
			r.unregister();
		}
	}

}
