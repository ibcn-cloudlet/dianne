package be.iminds.iot.dianne.builder;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;

import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.osgi.framework.BundleContext;
import org.osgi.framework.Constants;
import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;

import be.iminds.iot.dianne.api.nn.module.dto.ModuleDTO;
import be.iminds.iot.dianne.api.nn.module.dto.ModuleInstanceDTO;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkDTO;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkInstanceDTO;
import be.iminds.iot.dianne.api.nn.platform.DiannePlatform;
import be.iminds.iot.dianne.api.repository.DianneRepository;
import be.iminds.iot.dianne.nn.util.DianneJSONConverter;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonPrimitive;


@Component(service = { javax.servlet.Servlet.class }, 
	property = { "alias:String=/dianne/deployer",
				 "osgi.http.whiteboard.servlet.pattern=/dianne/deployer",
				 "aiolos.proxy=false" }, 
	immediate = true)
public class DianneDeployer extends HttpServlet {

	//public static final UUID UI_NN_ID = UUID.randomUUID();

	// mapping from string to UUID
	private Map<String, UUID> runtimeUUIDs = new HashMap<String, UUID>();
	private Map<UUID, String> runtimeNames = new HashMap<UUID, String>();

	private DianneRepository repository;
	private DiannePlatform platform;
	
	@Activate
	public void activate(BundleContext context){
		// hard coded mapping of known UUIDs to human readable names
		// TODO read this from config file or something?
		registerRuntimeName(UUID.fromString("00000000-0000-0000-0000-000000000000"), "Laptop");
		registerRuntimeName(UUID.fromString("00000000-0000-0000-0000-000000000001"), "Raspberry Pi");
		registerRuntimeName(UUID.fromString("00000000-0000-0000-0000-000000000002"), "Server");
		registerRuntimeName(UUID.fromString("00000000-0000-0000-0000-000000000003"), "Intel Edison");
		registerRuntimeName(UUID.fromString("00000000-0000-0000-0000-000000000004"), "nVidia Jetson");
		registerRuntimeName(UUID.fromString("00000000-0000-0000-0000-000000000005"), "GPU Server");
		registerRuntimeName(UUID.fromString(context.getProperty(Constants.FRAMEWORK_UUID)), "Local");
	}
	
	private void registerRuntimeName(UUID runtimeId, String runtimeName){
		runtimeNames.put(runtimeId, runtimeName);
		runtimeUUIDs.put(runtimeName, runtimeId);
	}
	
	@Reference
	void setDianne(DiannePlatform d){
		platform = d;
	}

	@Reference
	void setDianneRepository(DianneRepository repo){
		this.repository = repo;
	}
	
	@Override
	protected void doPost(HttpServletRequest request, HttpServletResponse response)
			throws ServletException, IOException {

		String action = request.getParameter("action");
		if(action.equals("deploy")){
			String id = request.getParameter("id");
			if(id==null){
				id = UUID.randomUUID().toString();
			}
			String name = request.getParameter("name");
			if(name==null){
				name = "unknown";
			}

			List<ModuleDTO> toDeploy = new ArrayList<ModuleDTO>();
			if(request.getParameter("modules")!=null){
				String modulesJsonString = request.getParameter("modules");
				NeuralNetworkDTO nn = DianneJSONConverter.parseJSON(modulesJsonString); 
				toDeploy.addAll(nn.modules);
			} else if(request.getParameter("module")!=null){
				String moduleJsonString = request.getParameter("module");
				ModuleDTO module = DianneJSONConverter.parseModuleJSON(moduleJsonString);
				toDeploy.add(module);
			}
				
			String target = request.getParameter("target");
			if(target == null){
				target = "Local"; // if no target specified, hard coded local target for now
			}
			UUID runtimeId = runtimeUUIDs.get(target);
			if(runtimeId==null){
				runtimeId = UUID.fromString(target);
			}
			
			try {
				List<ModuleInstanceDTO> moduleInstances = platform.deployModules(UUID.fromString(id), name, toDeploy, runtimeId);

				// return json object with deployment
				JsonObject result = new JsonObject();
				JsonObject deployment = deploymentToJSON(moduleInstances);
				result.add("id", new JsonPrimitive(id));
				result.add("deployment", deployment);
				try {
					response.getWriter().write(result.toString());
					response.getWriter().flush();
				} catch(IOException e){}
			
			} catch(InstantiationException e){
				System.out.println("Failed to deploy modules: "+e.getMessage());
			}
				
		} else if(action.equals("undeploy")){
			String id = request.getParameter("id");
			String moduleId = request.getParameter("moduleId");
			if(id!=null){
				NeuralNetworkInstanceDTO nn = platform.getNeuralNetworkInstance(UUID.fromString(id));
				if(nn!=null){
					ModuleInstanceDTO moduleInstance = nn.modules.get(UUID.fromString(moduleId));
					platform.undeployModules(Collections.singletonList(moduleInstance));
					
					response.getWriter().write(new JsonPrimitive(id).toString());
					response.getWriter().flush();
				}
			}
		} else if(action.equals("targets")){
			JsonArray targets = new JsonArray();
			
			List<UUID> runtimes = platform.getRuntimes();
			runtimes.stream()
					.map(runtimeId -> runtimeNames.get(runtimeId)==null ? runtimeId.toString() : runtimeNames.get(runtimeId))
					.forEach(runtimeName -> targets.add(new JsonPrimitive(runtimeName)));
			
			response.getWriter().write(targets.toString());
			response.getWriter().flush();
		} else if("recover".equals(action)){
			String id = request.getParameter("id");
			if(id==null){
				// list all options
				JsonArray nns = new JsonArray();
				platform.getNeuralNetworkInstances().stream().forEach(
						nni -> { JsonObject o = new JsonObject();
								o.add("id", new JsonPrimitive(nni.id.toString()));
								o.add("name", new JsonPrimitive(nni.name));
								if(nni.description!=null){
									o.add("description", new JsonPrimitive(nni.description));
								}
								nns.add(o);
						});
				response.getWriter().write(nns.toString());
				response.getWriter().flush();
				
			} else {
				NeuralNetworkInstanceDTO nni = platform.getNeuralNetworkInstance(UUID.fromString(id));
				if(nni==null){
					System.out.println("Failed to recover "+nni.id+" , instance not found");
					return;
				}
				
				try {
					response.getWriter().write("{\"nn\":");
					NeuralNetworkDTO nn = repository.loadNeuralNetwork(nni.name);
					String s = DianneJSONConverter.toJsonString(nn); 
					response.getWriter().write(s);
					response.getWriter().write(", \"layout\":");
					String layout = repository.loadLayout(nni.name);
					response.getWriter().write(layout);
					response.getWriter().write(", \"deployment\":");
					JsonObject deployment = deploymentToJSON(nni.modules.values());
					response.getWriter().write(deployment.toString());
					response.getWriter().write(", \"id\":");
					response.getWriter().write("\""+id+"\"");
					response.getWriter().write("}");
					response.getWriter().flush();
				} catch(Exception e){
					System.out.println("Failed to recover "+nni.name+" "+nni.id);
				}
			}
		}
		
	}
	
	private JsonObject deploymentToJSON(Collection<ModuleInstanceDTO> moduleInstances){
		JsonObject deployment = new JsonObject();
		for(ModuleInstanceDTO moduleInstance : moduleInstances){
			String runtime = runtimeNames.get(moduleInstance.runtimeId);
			if(runtime==null){
				runtime = moduleInstance.runtimeId.toString();
			}
			deployment.add(moduleInstance.moduleId.toString(), new JsonPrimitive(runtime));
		}
		return deployment;
	}
}
