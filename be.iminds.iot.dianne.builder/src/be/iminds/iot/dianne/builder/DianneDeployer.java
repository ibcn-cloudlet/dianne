package be.iminds.iot.dianne.builder;

import java.io.IOException;
import java.io.Writer;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Map.Entry;
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
import org.osgi.service.component.annotations.ReferenceCardinality;
import org.osgi.service.component.annotations.ReferencePolicy;

import be.iminds.iot.dianne.api.nn.module.dto.ModuleDTO;
import be.iminds.iot.dianne.api.nn.module.dto.ModuleInstanceDTO;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkDTO;
import be.iminds.iot.dianne.api.nn.runtime.ModuleManager;
import be.iminds.iot.dianne.nn.util.DianneJSONConverter;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonPrimitive;

@Component(service = { javax.servlet.Servlet.class }, 
	property = { "alias:String=/dianne/deployer","aiolos.proxy=false" }, 
	immediate = true)
public class DianneDeployer extends HttpServlet {

	public static final UUID UI_NN_ID = UUID.randomUUID();
	
	// this frameworks uuid
	private UUID frameworkId;
	
	// mapping from string to UUID
	private Map<String, UUID> runtimeUUIDs = Collections.synchronizedMap(new HashMap<String, UUID>());
	private Map<UUID, String> runtimeNames = Collections.synchronizedMap(new HashMap<UUID, String>());

	// mapping of UUID to ModuleManager
	private Map<UUID, ModuleManager> runtimes = Collections.synchronizedMap(new HashMap<UUID, ModuleManager>());
	
	private Map<UUID, UUID> deployment = Collections.synchronizedMap(new HashMap<UUID, UUID>());
	
	@Activate
	public void activate(BundleContext context){
		this.frameworkId = UUID.fromString(context.getProperty(Constants.FRAMEWORK_UUID));
	}
	
	@Reference(cardinality=ReferenceCardinality.AT_LEAST_ONE, 
			policy=ReferencePolicy.DYNAMIC)
	public void addModuleManager(ModuleManager m, Map<String, Object> properties){
		String name = (String) properties.get("aiolos.framework.uuid");
		UUID uuid = frameworkId;
		if(name!=null){
			uuid = UUID.fromString(name);
		}

		if(name == null){
			name = "localhost";
		} else if(name.equals("00000000-0000-0000-0000-000000000000")){
			name = "Laptop";
		} else if(name.equals("00000000-0000-0000-0000-000000000001")){
			// some hard coded values for demo
			name = "Raspberry Pi";
		} else if(name.equals("00000000-0000-0000-0000-000000000002")){
			// some hard coded values for demo
			name = "Server";
		} else if(name.equals("00000000-0000-0000-0000-000000000003")){
			// some hard coded values for demo
			name = "Intel Edison";
		} else if(name.equals("00000000-0000-0000-0000-000000000004")){
			// some hard coded values for demo
			name = "nVidia Jetson";
		} else if(name.equals("00000000-0000-0000-0000-000000000005")){
			// some hard coded values for demo
			name = "GPU Server";
		} else {
			// shorten it a bit TODO use human readable name
			name = name.substring(name.lastIndexOf('-')+1);
		}
		runtimeUUIDs.put(name, uuid);
		runtimeNames.put(uuid, name);
		runtimes.put(uuid, m);
	}
	
	public void removeModuleManager(ModuleManager m, Map<String, Object> properties){
		UUID uuid = UUID.fromString((String) properties.get("aiolos.framework.uuid")); 
		runtimes.remove(uuid);
		String name = runtimeNames.remove(uuid);
		runtimeUUIDs.remove(name);
	}
	

	@Override
	protected void doPost(HttpServletRequest request, HttpServletResponse response)
			throws ServletException, IOException {

		String action = request.getParameter("action");
		if(action.equals("deploy")){
			if(request.getParameter("modules")!=null){
				String modulesJsonString = request.getParameter("modules");
				String target = request.getParameter("target");
				if(target == null){
					target = "local"; // if no target specified, hard coded local target for now
				}
				NeuralNetworkDTO nn = DianneJSONConverter.parseJSON(modulesJsonString); 
					
				for(ModuleDTO module : nn.modules){
					deployModule(module, target);
				}
				// TODO only return deployment of deployed modules?
				returnDeployment(response.getWriter());
			} else if(request.getParameter("module")!=null){
				String moduleJsonString = request.getParameter("module");
				String target = request.getParameter("target");
				ModuleDTO module = DianneJSONConverter.parseModuleJSON(moduleJsonString); 
				deployModule(module, target);
				// TODO only return deployment of deployed modules?
				returnDeployment(response.getWriter());
			}
		} else if(action.equals("undeploy")){
			String id = request.getParameter("id");
			if(id!=null){
				undeployModule(UUID.fromString(id));
				response.getWriter().write(new JsonPrimitive(id).toString());
				response.getWriter().flush();
			}
		} else if(action.equals("targets")){
			JsonArray targets = new JsonArray();
			synchronized(runtimes){
				for(String id : runtimeNames.values()){
					targets.add(new JsonPrimitive(id));
				}
			}
			response.getWriter().write(targets.toString());
			response.getWriter().flush();
		}
		
	}
	
	private void deployModule(ModuleDTO module, String target){
		UUID deployTo = runtimeUUIDs.get(target);
		
		UUID migrateFrom = null;
		if(deployment.containsKey(module.id)){
			// already deployed... TODO exception or something?
			migrateFrom = deployment.get(module.id); 
			if(deployTo.equals(migrateFrom)){
				return;
			}
		}
		
		try {
			ModuleManager runtime = runtimes.get(deployTo);
			if(runtime!=null){
				runtime.deployModule(module, UI_NN_ID);
				deployment.put(module.id, deployTo);
			}
			
			// when migrating, undeploy module from previous
			if(migrateFrom!=null){
				runtime = runtimes.get(migrateFrom);
				if(runtime!=null){
					runtime.undeployModule(new ModuleInstanceDTO(module.id, DianneDeployer.UI_NN_ID, migrateFrom));
				}
			}
		} catch (Exception e) {
			System.err.println("Failed to deploy module "+module.id);
			e.printStackTrace();
		}
	}
	
	private void undeployModule(UUID id){
		try {
			UUID target = deployment.remove(id);
			if(target!=null){
				ModuleManager runtime = runtimes.get(target);
				if(runtime!=null){
					runtime.undeployModule(new ModuleInstanceDTO(id, DianneDeployer.UI_NN_ID, target));
				}
			}
		} catch (Exception e) {
			System.err.println("Failed to deploy module "+id);
			e.printStackTrace();
		}
	}
	
	private void returnDeployment(Writer writer){
		JsonObject result = new JsonObject();
		synchronized(deployment){
			for(Iterator<Entry<UUID,UUID>> it = deployment.entrySet().iterator();it.hasNext();){
				Entry<UUID, UUID> e = it.next();
				String name = runtimeNames.get(e.getValue());
				result.add(e.getKey().toString(), new JsonPrimitive(name));
			}
		}
		try {
			writer.write(result.toString());
			writer.flush();
		} catch(IOException e){}
	}
}
