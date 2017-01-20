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

import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;

import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkDTO;
import be.iminds.iot.dianne.api.repository.DianneRepository;
import be.iminds.iot.dianne.nn.util.DianneJSONConverter;

import com.google.gson.JsonArray;
import com.google.gson.JsonPrimitive;

@Component(service = { javax.servlet.Servlet.class }, 
	property = { "alias:String=/dianne/load",
		 		 "osgi.http.whiteboard.servlet.pattern=/dianne/load",
		 		 "aiolos.proxy=false" }, 
	immediate = true)
public class DianneLoader extends HttpServlet {
	
	private static final long serialVersionUID = 1L;
	
	private DianneRepository repository;
	
	@Reference
	void setDianneRepository(DianneRepository repo){
		this.repository = repo;
	}
	
	@Override
	protected void doPost(HttpServletRequest request, HttpServletResponse response)
			throws ServletException, IOException {
		response.setContentType("application/json");

		String action = request.getParameter("action");
		if("list".equals(action)){
			JsonArray names = new JsonArray();
			for(String name : repository.availableNeuralNetworks()){
				names.add(new JsonPrimitive(name));
			}
			response.getWriter().write(names.toString());
			response.getWriter().flush();
		} else if("load".equals(action)){
			String name = request.getParameter("name");
			
			response.getWriter().write("{\"nn\":");
			NeuralNetworkDTO nn = repository.loadNeuralNetwork(name);
			String s = DianneJSONConverter.toJsonString(nn); 
			response.getWriter().write(s);
			response.getWriter().write(", \"layout\":");
			String layout = repository.loadLayout(name);
			response.getWriter().write(layout);
			response.getWriter().write("}");
			response.getWriter().flush();
		}
	}
}
