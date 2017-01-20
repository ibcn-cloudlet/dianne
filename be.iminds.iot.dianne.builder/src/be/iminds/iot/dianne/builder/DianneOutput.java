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
import java.util.Collections;
import java.util.List;
import java.util.UUID;

import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;
import org.osgi.service.component.annotations.ReferenceCardinality;
import org.osgi.service.component.annotations.ReferencePolicy;

import be.iminds.iot.dianne.api.io.OutputDescription;
import be.iminds.iot.dianne.api.io.DianneOutputs;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonPrimitive;

@Component(service = { javax.servlet.Servlet.class }, 
	property = { "alias:String=/dianne/output",
				 "osgi.http.whiteboard.servlet.pattern=/dianne/output",
				 "aiolos.proxy=false" }, 
	immediate = true)
public class DianneOutput extends HttpServlet {
	
	private static final long serialVersionUID = 1L;
	
	private List<DianneOutputs> outputs = Collections.synchronizedList(new ArrayList<DianneOutputs>());
	
	@Reference(cardinality=ReferenceCardinality.AT_LEAST_ONE, 
			policy=ReferencePolicy.DYNAMIC)
	void addOutputs(DianneOutputs mgr){
		this.outputs.add(mgr);
	}
	
	void removeOutputs(DianneOutputs mgr){
		this.outputs.remove(mgr);
	}
	
	@Override
	protected void doPost(HttpServletRequest request, HttpServletResponse response)
			throws ServletException, IOException {
		response.setContentType("application/json");

		String action = request.getParameter("action");
		if("available-outputs".equals(action)){
			JsonArray availableOutputs = new JsonArray();
			synchronized(outputs){
				for(DianneOutputs o : outputs){
					for(OutputDescription output : o.getAvailableOutputs()){
						JsonObject ob = new JsonObject();
						ob.add("name", new JsonPrimitive(output.getName()));
						ob.add("type", new JsonPrimitive(output.getType()));
						availableOutputs.add(ob);
					}
				}
			}
			response.getWriter().write(availableOutputs.toString());
			response.getWriter().flush();
		} else if("setoutput".equals(action)){
			String nnId = request.getParameter("nnId");
			String outputId = request.getParameter("outputId");
			String output = request.getParameter("output");
			// TODO only forward to applicable outputmgr?
			synchronized(outputs){
				for(DianneOutputs o : outputs){
					o.setOutput(UUID.fromString(nnId), UUID.fromString(outputId), output);
				}
			}
		} else if("unsetoutput".equals(action)){
			String nnId = request.getParameter("nnId");
			String outputId = request.getParameter("outputId");
			String output = request.getParameter("output");
			// TODO only forward to applicable outputmgr?
			synchronized(outputs){
				for(DianneOutputs o : outputs){
					o.unsetOutput(UUID.fromString(nnId), UUID.fromString(outputId), output);
				}
			}
		}
	}
}
