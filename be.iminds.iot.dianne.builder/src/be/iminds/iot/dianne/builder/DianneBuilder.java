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
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;
import org.osgi.service.component.annotations.ReferenceCardinality;
import org.osgi.service.component.annotations.ReferencePolicy;
import org.osgi.service.http.HttpService;

import be.iminds.iot.dianne.api.nn.module.dto.ModulePropertyDTO;
import be.iminds.iot.dianne.api.nn.module.dto.ModuleTypeDTO;
import be.iminds.iot.dianne.api.nn.module.factory.ModuleFactory;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonPrimitive;

@Component(service={javax.servlet.Servlet.class},
	property={"alias:String=/dianne/builder",
		 	  "osgi.http.whiteboard.servlet.pattern=/dianne/builder",
			  "aiolos.proxy=false"},
	immediate=true)
public class DianneBuilder extends HttpServlet {

	private static final long serialVersionUID = 1L;
	
	private List<ModuleFactory> factories = Collections.synchronizedList(new ArrayList<ModuleFactory>());
	
	@Reference(cardinality=ReferenceCardinality.AT_LEAST_ONE, 
			policy=ReferencePolicy.DYNAMIC)
	void addModuleFactory(ModuleFactory factory){
		factories.add(factory);
	}
	
	void removeModuleFactory(ModuleFactory factory){
		factories.remove(factory);
	}
	
	@Reference
	void setHttpService(HttpService http){
		try {
			// TODO How to register resources with whiteboard pattern?
			http.registerResources("/dianne/ui/builder", "res", null);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	@Override
	protected void doGet(HttpServletRequest req, HttpServletResponse resp) throws ServletException, IOException {
		resp.sendRedirect("/dianne/ui/builder/builder.html");
	}
	
	@Override
	protected void doPost(HttpServletRequest request, HttpServletResponse response)
			throws ServletException, IOException {
		response.setContentType("application/json");
		String action = request.getParameter("action");

		if(action.equals("available-modules")){
			getAvailableModules(response.getWriter());
		} else if(action.equals("module-properties")){
			String type = request.getParameter("type");
			getModuleProperties(type, response.getWriter());
		} 
	}
	
	private void getAvailableModules(PrintWriter writer){
		List<ModuleTypeDTO> moduleTypes = new ArrayList<ModuleTypeDTO>();
		synchronized(factories){
			for(ModuleFactory f : factories){
				moduleTypes.addAll(f.getAvailableModuleTypes());
			}
		}
		
		JsonArray jsonModules = new JsonArray();
		for(ModuleTypeDTO moduleType : moduleTypes){
			JsonObject jsonModule = new JsonObject();
			jsonModule.add("type", new JsonPrimitive(moduleType.type));
			jsonModule.add("category", new JsonPrimitive(moduleType.category));
			if(moduleType.trainable){
				jsonModule.add("trainable", new JsonPrimitive(true));
			}
			jsonModules.add(jsonModule);
		}
		writer.write(jsonModules.toString());
		writer.flush();
	}
	
	private void getModuleProperties(String type, PrintWriter writer){
		ModuleTypeDTO module = null;
		synchronized(factories){
			for(ModuleFactory f : factories){
				module = f.getModuleType(type);
				if(module!=null)
					break;
			}
		}
		
		if(module==null){
			// return;
		}
		
		JsonArray jsonProperties = new JsonArray();
		for(ModulePropertyDTO p : module.properties){
			JsonObject jsonProperty = new JsonObject();
			jsonProperty.addProperty("id", p.id);
			jsonProperty.addProperty("name", p.name);
			
			jsonProperties.add(jsonProperty);
		}
		
		writer.write(jsonProperties.toString());
		writer.flush();
	}
}
