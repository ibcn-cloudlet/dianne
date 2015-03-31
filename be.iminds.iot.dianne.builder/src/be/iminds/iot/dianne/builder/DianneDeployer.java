package be.iminds.iot.dianne.builder;

import java.io.IOException;
import java.io.Writer;
import java.util.Collections;
import java.util.Dictionary;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.UUID;

import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;
import org.osgi.service.component.annotations.ReferenceCardinality;
import org.osgi.service.component.annotations.ReferencePolicy;

import be.iminds.iot.dianne.api.nn.runtime.ModuleManager;
import be.iminds.iot.dianne.nn.runtime.util.DianneJSONParser;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonPrimitive;

@Component(service = { javax.servlet.Servlet.class }, 
	property = { "alias:String=/dianne/deployer","aiolos.proxy=false" }, 
	immediate = true)
public class DianneDeployer extends HttpServlet {

	private Map<String, ModuleManager> runtimes = Collections.synchronizedMap(new HashMap<String, ModuleManager>());
	
	private Map<String, String> deployment = Collections.synchronizedMap(new HashMap<String, String>());
	
	@Reference(cardinality=ReferenceCardinality.AT_LEAST_ONE, 
			policy=ReferencePolicy.DYNAMIC)
	public void addModuleManager(ModuleManager m, Map<String, Object> properties){
		String uuid = (String) properties.get("aiolos.framework.uuid"); 
		if(uuid==null){
			uuid = "Laptop"; // TODO for now just fixed item for local runtime
		} else if(uuid.equals("00000000-0000-0000-0000-000000000001")){
			// some hard coded values for demo
			uuid = "Raspberry Pi";
		} else if(uuid.equals("00000000-0000-0000-0000-000000000002")){
			// some hard coded values for demo
			uuid = "Laptop";
		} else if(uuid.equals("00000000-0000-0000-0000-000000000003")){
			// some hard coded values for demo
			uuid = "Intel Edison";
		} else {
			// shorten it a bit TODO use human readable name
			uuid = uuid.substring(0, uuid.indexOf('-'));
		}
		runtimes.put(uuid, m);
	}
	
	public void removeModuleManager(ModuleManager m, Map<String, Object> properties){
		String uuid = (String) properties.get("aiolos.framework.uuid"); 
		if(uuid==null){
			uuid = "Laptop"; // TODO for now just fixed item for local runtime
		} else if(uuid.equals("00000000-0000-0000-0000-000000000001")){
			// some hard coded values for demo
			uuid = "Raspberry Pi";
		} else if(uuid.equals("00000000-0000-0000-0000-000000000002")){
			// some hard coded values for demo
			uuid = "Laptop";
		} else if(uuid.equals("00000000-0000-0000-0000-000000000003")){
			// some hard coded values for demo
			uuid = "Intel Edison";
		} else {
			// shorten it a bit TODO use human readable name
			uuid = uuid.substring(0, uuid.indexOf('-'));
		}
		runtimes.remove(uuid);
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
				List<Dictionary<String, Object>> modules = DianneJSONParser.parseJSON(modulesJsonString); 
						
				for(Dictionary<String, Object> module : modules){
					deployModule(module, target);
				}
				// TODO only return deployment of deployed modules?
				returnDeployment(response.getWriter());
			} else if(request.getParameter("module")!=null){
				String moduleJsonString = request.getParameter("module");
				String target = request.getParameter("target");
				Dictionary<String, Object> module = DianneJSONParser.parseModuleJSON(moduleJsonString); 
				deployModule(module, target);
				// TODO only return deployment of deployed modules?
				returnDeployment(response.getWriter());
			}
		} else if(action.equals("undeploy")){
			String id = request.getParameter("id");
			if(id!=null){
				undeployModule(id);
				response.getWriter().write(new JsonPrimitive(id).toString());
				response.getWriter().flush();
			}
		} else if(action.equals("targets")){
			JsonArray targets = new JsonArray();
			synchronized(runtimes){
				for(String id : runtimes.keySet()){
					targets.add(new JsonPrimitive(id));
				}
			}
			response.getWriter().write(targets.toString());
			response.getWriter().flush();
		}
		
	}
	
	private void deployModule(Dictionary<String, Object> config, String target){
		String id = (String)config.get("module.id");
		String migrateFrom = null;
		if(deployment.containsKey(id)){
			// already deployed... TODO exception or something?
			migrateFrom = deployment.get(id); 
			if(target.equals(migrateFrom)){
				return;
			}
		}
		
		try {
			ModuleManager runtime = runtimes.get(target);
			if(runtime!=null){
				runtime.deployModule(config);
				deployment.put(id, target);
			}
			
			// when migrating, undeploy module from previous
			if(migrateFrom!=null){
				runtime = runtimes.get(migrateFrom);
				if(runtime!=null){
					runtime.undeployModule(UUID.fromString(id));
				}
			}
		} catch (Exception e) {
			System.err.println("Failed to deploy module "+id);
			e.printStackTrace();
		}
	}
	
	private void undeployModule(String id){
		try {
			String target = deployment.remove(id);
			if(target!=null){
				ModuleManager runtime = runtimes.get(target);
				if(runtime!=null){
					runtime.undeployModule(UUID.fromString(id));
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
			for(Iterator<Entry<String,String>> it = deployment.entrySet().iterator();it.hasNext();){
				Entry<String, String> e = it.next();
				result.add(e.getKey(), new JsonPrimitive(e.getValue()));
			}
		}
		try {
			writer.write(result.toString());
			writer.flush();
		} catch(IOException e){}
	}
}
