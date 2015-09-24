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

import be.iminds.iot.dianne.api.io.InputDescription;
import be.iminds.iot.dianne.api.io.DianneInputs;
import be.iminds.iot.dianne.api.nn.module.Input;
import be.iminds.iot.dianne.tensor.TensorFactory;
import be.iminds.iot.things.api.Thing;
import be.iminds.iot.things.api.camera.Camera;
import be.iminds.iot.things.api.camera.CameraListener;

@Component(immediate=true)
public class CameraInputs implements DianneInputs {

	private BundleContext context;
	
	private TensorFactory factory;
	
	private Map<String, UUID> cameraIds = Collections.synchronizedMap(new HashMap<String, UUID>());
	private Map<UUID, Camera> cameras = Collections.synchronizedMap(new HashMap<UUID, Camera>());

	// these are mapped by the string moduleId-nnId  ... TODO use ModuleInstanceDTO for this?
	private Map<String, Input> inputs = Collections.synchronizedMap(new HashMap<String, Input>());
	private Map<String, ServiceRegistration> registrations =  Collections.synchronizedMap(new HashMap<String, ServiceRegistration>());
	
	
	@Activate
	void activate(BundleContext context){
		this.context = context;
	}
	
	@Reference
	void setTensorFactory(TensorFactory factory){
		this.factory = factory;
	}
	
	@Reference(
			cardinality=ReferenceCardinality.MULTIPLE, 
			policy=ReferencePolicy.DYNAMIC)
	void addCamera(Camera c, Map<String, Object> properties){
		UUID id = (UUID) properties.get(Thing.ID);
		String service = (String) properties.get(Thing.SERVICE);
		
		String[] parts = service.split("_");
		String filtered = "Webcam "+parts[1];
		// TODO only handle local thing services?
		// TODO use id-service combo?
		cameraIds.put(filtered, id);
		cameras.put(id, c);
		
	}
	
	void removeCamera(Camera c, Map<String, Object> properties){
		UUID id = (UUID) properties.get(Thing.ID);
		String service = (String) properties.get(Thing.SERVICE);
		
		String[] parts = service.split("_");
		String filtered = parts[1];
		cameraIds.remove(filtered);
		cameras.remove(id);
		
	}
	
	@Reference(cardinality=ReferenceCardinality.MULTIPLE, 
			policy=ReferencePolicy.DYNAMIC)
	void addInput(Input i, Map<String, Object> properties){
		String moduleId = (String)properties.get("module.id");
		String nnId = (String)properties.get("module.id");
		String id = nnId+":"+moduleId;
		inputs.put(id, i);
	}
	
	void removeInput(Input i, Map<String, Object> properties){
		String moduleId = (String)properties.get("module.id");
		String nnId = (String)properties.get("module.id");
		String id = nnId+":"+moduleId;
		inputs.remove(UUID.fromString(id));
	}
	
	@Override
	public List<InputDescription> getAvailableInputs() {
		ArrayList<InputDescription> inputs = new ArrayList<InputDescription>();
		synchronized(cameraIds){
			for(String id : cameraIds.keySet()){
				inputs.add(new InputDescription(id, "Camera"));
			}
		}
		return inputs;
	}

	@Override
	public void setInput(UUID inputId, UUID nnId, String input) {
		String id = nnId.toString()+":"+inputId.toString();
		
		UUID cameraId = cameraIds.get(input);
		if(cameraId!=null){
			Camera camera = cameras.get(cameraId);
			if(camera!=null){
				camera.setFramerate(0.1f); // low framerate for overfeat
				camera.start(320, 240, Camera.Format.RGB);
			}
			
			CameraInput i = new CameraInput(factory, inputs.get(id), 320, 240, 3);
			Dictionary<String, Object> properties = new Hashtable<String, Object>();
			properties.put(CameraListener.CAMERA_ID, cameraId.toString());
			properties.put("aiolos.unique", true);
			ServiceRegistration r = context.registerService(CameraListener.class.getName(), i, properties);
			// TODO only works if outputId only forwards to one output
			registrations.put(id, r);
		}
	}

	@Override
	public void unsetInput(UUID inputId, UUID nnId, String input) {
		String id = nnId.toString()+":"+inputId.toString();
		
		UUID cameraId = cameraIds.get(input);
		if(cameraId!=null){
			Camera camera = cameras.get(cameraId);
			if(camera!=null){
				camera.stop();
			}
		}
		
		ServiceRegistration r = registrations.remove(id);
		if(r!=null){
			r.unregister();
		}
	}

}
