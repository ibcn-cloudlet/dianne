package be.iminds.iot.dianne.builder;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Dictionary;
import java.util.Hashtable;
import java.util.List;
import java.util.Map.Entry;

import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;
import org.osgi.service.component.annotations.ReferenceCardinality;
import org.osgi.service.component.annotations.ReferencePolicy;
import org.osgi.service.http.HttpService;

import be.iminds.iot.dianne.nn.runtime.ModuleManager;

import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

@Component(service = { javax.servlet.Servlet.class }, 
	property = { "alias:String=/dianne/deployer" }, 
	immediate = true)
public class DianneDeployer extends HttpServlet {

	private List<ModuleManager> runtimes = Collections.synchronizedList(new ArrayList<ModuleManager>());
	
	@Reference(cardinality=ReferenceCardinality.AT_LEAST_ONE, 
			policy=ReferencePolicy.DYNAMIC)
	public void addModuleManager(ModuleManager m){
		runtimes.add(m);
	}
	
	public void removeModuleManager(ModuleManager m){
		runtimes.remove(m);
	}
	
	@Reference
	public void setHttpService(HttpService http){
		try {
			// Use manual registration - problems with whiteboard
			http.registerServlet("/dianne/deployer", this, null, null);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	@Override
	protected void doPost(HttpServletRequest request, HttpServletResponse response)
			throws ServletException, IOException {

		System.out.println("Deploy!");
		
		String modulesJsonString = request.getParameter("modules");
		JsonObject modulesJson = new JsonParser().parse(modulesJsonString).getAsJsonObject();
		
		for(Entry<String, JsonElement> module : modulesJson.entrySet()){
			System.out.println(module.getKey());
			
			JsonObject moduleJson = (JsonObject) module.getValue();
			
			// new module
			Dictionary<String, Object> properties = new Hashtable<String, Object>();
			
			String id = moduleJson.get("id").getAsString();
			String type = moduleJson.get("type").getAsString();
			
			properties.put("module.id", id);
			properties.put("module.type", type);
			
			if(moduleJson.has("next")){
				properties.put("module.next", moduleJson.get("next").getAsString());
			}
			if(moduleJson.has("prev")){
				properties.put("module.prev", moduleJson.get("prev").getAsString());
			}

			// key prefix
			String prefix = "module."+type.toLowerCase()+"."; 
				
			for(Entry<String, JsonElement> property : moduleJson.entrySet()){
				String key = property.getKey();
				if(key.equals("id")
					|| key.equals("type")
					|| key.equals("prev")
					|| key.equals("next")){
					continue;
					// this is only for module-specific properties
				}
				// TODO already infer type here?
				properties.put(prefix+property.getKey(), property.getValue().getAsString());
			}
			
			// for now deploy all modules on one runtime
			try {
				runtimes.get(0).deployModule(properties);
			} catch (InstantiationException e) {
				System.err.println("Failed to deploy module "+id);
				e.printStackTrace();
			}
		}
		
	}
}
