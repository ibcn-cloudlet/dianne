/*******************************************************************************
 * DIANNE  - Framework for distributed artificial neural networks
 * Copyright (C) 2015  iMinds - IBCN - UGent
 *
 * This file is part of DIANNE.
 *
 * DIANNE is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * Contributors:
 *     Tim Verbelen, Steven Bohez
 *******************************************************************************/
package be.iminds.iot.dianne.builder;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.UUID;

import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.osgi.framework.BundleContext;
import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonPrimitive;

import be.iminds.iot.dianne.api.nn.module.dto.ModuleDTO;
import be.iminds.iot.dianne.api.nn.module.dto.ModuleInstanceDTO;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkDTO;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkInstanceDTO;
import be.iminds.iot.dianne.api.nn.platform.DiannePlatform;
import be.iminds.iot.dianne.api.repository.DianneRepository;
import be.iminds.iot.dianne.nn.util.DianneJSONConverter;


@Component(service = { javax.servlet.Servlet.class }, 
	property = { "alias:String=/dianne/deployer",
				 "osgi.http.whiteboard.servlet.pattern=/dianne/deployer",
				 "aiolos.proxy=false" }, 
	immediate = true)
public class DianneDeployer extends HttpServlet {

	private static final long serialVersionUID = 1L;
	
	private DianneRepository repository;
	private DiannePlatform platform;
	
	@Activate
	public void activate(BundleContext context){
	}
	
	@Reference
	void setDiannePlatform(DiannePlatform p){
		platform = p;
	}

	@Reference
	void setDianneRepository(DianneRepository repo){
		this.repository = repo;
	}
	
	@Override
	protected void doPost(HttpServletRequest request, HttpServletResponse response)
			throws ServletException, IOException {
		response.setContentType("application/json");

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
			String[] tags = null;
			String t = request.getParameter("tags");
			if(t != null && !t.isEmpty()){
				tags = t.split(",");
			}
			
			List<ModuleDTO> toDeploy = new ArrayList<ModuleDTO>();
			if(request.getParameter("modules")!=null){
				String modulesJsonString = request.getParameter("modules");
				NeuralNetworkDTO nn = DianneJSONConverter.parseJSON(modulesJsonString); 
				toDeploy.addAll(nn.modules.values());
			} else if(request.getParameter("module")!=null){
				String moduleJsonString = request.getParameter("module");
				ModuleDTO module = DianneJSONConverter.parseModuleJSON(moduleJsonString);
				toDeploy.add(module);
			}
				
			try {
				String target = request.getParameter("target");
				if(target==null){
					throw new Exception("No DIANNE runtime selected to deploy to");
				}
				UUID runtimeId = UUID.fromString(target);
				
				UUID nnId = UUID.fromString(id);
				List<ModuleInstanceDTO> moduleInstances = platform.deployModules(nnId, name, toDeploy, runtimeId, tags);
				
				// return json object with deployment
				JsonObject result = new JsonObject();
				JsonObject deployment = deploymentToJSON(moduleInstances);
				result.add("id", new JsonPrimitive(id));
				result.add("deployment", deployment);
				try {
					response.getWriter().write(result.toString());
					response.getWriter().flush();
				} catch(IOException e){}
			
			} catch(Exception ex){
				System.out.println("Failed to deploy modules: "+ex.getMessage());
				JsonObject result = new JsonObject();
				result.add("error", new JsonPrimitive("Failed to deploy modules: "+ex.getMessage()));
				try {
					response.getWriter().write(result.toString());
					response.getWriter().flush();
				} catch(IOException e){}
			} 
				
		} else if(action.equals("undeploy")){
			String id = request.getParameter("id");
			String moduleId = request.getParameter("moduleId");
			if(id!=null){
				NeuralNetworkInstanceDTO nn = platform.getNeuralNetworkInstance(UUID.fromString(id));
				if(nn!=null){
					ModuleInstanceDTO moduleInstance = nn.modules.get(UUID.fromString(moduleId));
					if(moduleInstance!=null)
						platform.undeployModules(Collections.singletonList(moduleInstance));
				
					
					response.getWriter().write(new JsonPrimitive(id).toString());
					response.getWriter().flush();
				}
			}
		} else if(action.equals("targets")){
			JsonArray targets = new JsonArray();
			
			Map<UUID, String> runtimes = platform.getRuntimes();
			runtimes.entrySet().stream()
				.forEach(e -> {
					JsonObject o = new JsonObject();
					o.add("id", new JsonPrimitive(e.getKey().toString()));
					o.add("name", new JsonPrimitive(e.getValue()));
					targets.add(o);
				});
			
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
					System.out.println("Failed to recover "+id+" , instance not found");
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
			String runtime = moduleInstance.runtimeId.toString();
			if(runtime==null){
				runtime = moduleInstance.runtimeId.toString();
			}
			deployment.add(moduleInstance.moduleId.toString(), new JsonPrimitive(runtime));
		}
		return deployment;
	}
}
