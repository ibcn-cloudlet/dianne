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
import java.util.Arrays;
import java.util.Collections;
import java.util.Dictionary;
import java.util.HashMap;
import java.util.Hashtable;
import java.util.List;
import java.util.Map;
import java.util.UUID;

import javax.servlet.AsyncContext;
import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.osgi.framework.BundleContext;
import org.osgi.framework.ServiceRegistration;
import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;
import org.osgi.service.component.annotations.ReferenceCardinality;
import org.osgi.service.component.annotations.ReferencePolicy;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.google.gson.JsonPrimitive;

import be.iminds.iot.dianne.api.io.DianneInputs;
import be.iminds.iot.dianne.api.io.InputDescription;
import be.iminds.iot.dianne.api.nn.module.ForwardListener;
import be.iminds.iot.dianne.api.nn.module.ModuleException;
import be.iminds.iot.dianne.tensor.Tensor;

@Component(service = { javax.servlet.Servlet.class }, 
	property = { "alias:String=/dianne/input",
		 	     "osgi.http.whiteboard.servlet.pattern=/dianne/input",
		 	     "osgi.http.whiteboard.servlet.asyncSupported:Boolean=true",
				 "aiolos.proxy=false" }, 
	immediate = true)
public class DianneInput extends HttpServlet {
	
	private List<DianneInputs> inputs = Collections.synchronizedList(new ArrayList<DianneInputs>());

	// Send event to UI when new input sample arrived
	private AsyncContext sse = null;
	private BundleContext context;
	private Map<UUID, ServiceRegistration> inputListeners = Collections.synchronizedMap(new HashMap<UUID, ServiceRegistration>());
	private JsonParser parser = new JsonParser();

	@Activate 
	public void activate(BundleContext context){
		this.context = context;
	}
	
	@Reference(cardinality=ReferenceCardinality.AT_LEAST_ONE, 
			policy=ReferencePolicy.DYNAMIC)
	void addInputs(DianneInputs mgr){
		this.inputs.add(mgr);
	}
	
	void removeInputs(DianneInputs mgr){
		this.inputs.remove(mgr);
	}
	
	@Override
	protected void doGet(HttpServletRequest request, HttpServletResponse response)
			throws ServletException, IOException {
		// write text/eventstream response
		response.setContentType("text/event-stream");
		response.setHeader("Cache-Control", "no-cache");
		response.setCharacterEncoding("UTF-8");
		response.addHeader("Connection", "keep-alive");
		
		sse = request.startAsync();
		sse.setTimeout(0); // no timeout => remove listener when error occurs.
	}
	
	@Override
	protected void doPost(HttpServletRequest request, HttpServletResponse response)
			throws ServletException, IOException {

		String action = request.getParameter("action");
		if("available-inputs".equals(action)){
			JsonArray availableInputs = new JsonArray();
			synchronized(inputs){
				for(DianneInputs i : inputs){
					for(InputDescription input : i.getAvailableInputs()){
						JsonObject o = new JsonObject();
						o.add("name", new JsonPrimitive(input.getName()));
						o.add("type", new JsonPrimitive(input.getType()));
						availableInputs.add(o);
					}
				}
			}
			response.getWriter().write(availableInputs.toString());
			response.getWriter().flush();
		} else if("setinput".equals(action)){
			String nnId = request.getParameter("nnId");
			String inputId = request.getParameter("inputId");
			String input = request.getParameter("input");

			synchronized(inputs){
				for(DianneInputs i : inputs){
					i.setInput(UUID.fromString(nnId), UUID.fromString(inputId), input);
				}
				
				ForwardListener inputListener = new ForwardListener(){
					@Override
					public void onForward(UUID moduleId, Tensor output, String... tags) {
						if(sse!=null){
							try {
								JsonObject data = new JsonObject();

								if(output.dim()==3){
									data.add("channels", new JsonPrimitive(output.dims()[0]));
									data.add("height", new JsonPrimitive(output.dims()[1]));
									data.add("width", new JsonPrimitive(output.dims()[2]));
								} else if(output.dim()==2) {
									data.add("channels", new JsonPrimitive(1));
									data.add("height", new JsonPrimitive(output.dims()[0]));
									data.add("width", new JsonPrimitive(output.dims()[1]));
								} else {
									data.add("size", new JsonPrimitive(output.size()));
								}
								if(tags!=null){
									JsonArray ta = new JsonArray();
									for(String t : tags){
										ta.add(new JsonPrimitive(t));
									}
									data.add("tags",ta);
								}
								
								data.add("data", parser.parse(Arrays.toString(output.get())));
								
								StringBuilder builder = new StringBuilder();
								builder.append("data: ").append(data.toString()).append("\n\n");
				
								PrintWriter writer = sse.getResponse().getWriter();
								writer.write(builder.toString());
								writer.flush();
							} catch(Exception e){}
						}
					}

					@Override
					public void onError(UUID moduleId, ModuleException e, String... tags) {
						e.printStackTrace();
					}
				};
				Dictionary<String, Object> properties = new Hashtable<String, Object>();
				properties.put("targets", new String[]{nnId+":"+inputId});
				properties.put("aiolos.unique", true);
				ServiceRegistration r = context.registerService(ForwardListener.class.getName(), inputListener, properties);
				inputListeners.put(UUID.fromString(inputId), r);
			}
		} else if("unsetinput".equals(action)){
			String nnId = request.getParameter("nnId");
			String inputId = request.getParameter("inputId");
			String input = request.getParameter("input");

			synchronized(inputs){
				for(DianneInputs i : inputs){
					i.unsetInput(UUID.fromString(nnId), UUID.fromString(inputId), input);
				}
			}
			
			ServiceRegistration r = inputListeners.get(UUID.fromString(inputId));
			if(r!=null){
				r.unregister();
			}
		}
	}
}
