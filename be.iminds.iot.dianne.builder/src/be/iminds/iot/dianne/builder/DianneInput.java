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
import java.util.Dictionary;
import java.util.HashMap;
import java.util.Hashtable;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.UUID;

import javax.servlet.AsyncContext;
import javax.servlet.AsyncEvent;
import javax.servlet.AsyncListener;
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
import com.google.gson.JsonPrimitive;

import be.iminds.iot.dianne.api.io.DianneInputs;
import be.iminds.iot.dianne.api.io.InputDescription;
import be.iminds.iot.dianne.api.nn.module.ForwardListener;
import be.iminds.iot.dianne.api.nn.module.ModuleException;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.util.JsonConverter;

@Component(service = { javax.servlet.Servlet.class }, 
	property = { "alias:String=/dianne/input",
		 	     "osgi.http.whiteboard.servlet.pattern=/dianne/input",
		 	     "osgi.http.whiteboard.servlet.asyncSupported:Boolean=true",
				 "aiolos.proxy=false" }, 
	immediate = true)
public class DianneInput extends HttpServlet {
	
	private static final long serialVersionUID = 1L;

	private List<DianneInputs> inputs = Collections.synchronizedList(new ArrayList<DianneInputs>());

	// Send event to UI when new input sample arrived
	private Map<String, List<AsyncContext>> sses = Collections.synchronizedMap(new HashMap<>());
	private BundleContext context;
	private Map<UUID, ServiceRegistration<ForwardListener>> inputListeners = Collections.synchronizedMap(new HashMap<>());
	private JsonConverter converter = new JsonConverter();

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
		
		// register forward listener for this
		final String name = request.getParameter("name");
		if(name != null){
			final AsyncContext sse = request.startAsync();
			sse.addListener(new AsyncListener() {
				@Override public void onComplete(AsyncEvent event) throws IOException {sses.get(name).remove(sse);}
			    @Override public void onTimeout(AsyncEvent event) throws IOException {sses.get(name).remove(sse);}
			    @Override public void onError(AsyncEvent event) throws IOException {sses.get(name).remove(sse);}
			    @Override public void onStartAsync(AsyncEvent event) throws IOException {}
			});
			sse.setTimeout(0); // no timeout => remove listener when error occurs.
			List<AsyncContext> list = sses.get(name);
			if(list == null){
				list = new ArrayList<>();
				sses.put(name, list);
			}
			list.add(sse);
		}
	}
	
	@Override
	protected void doPost(HttpServletRequest request, HttpServletResponse response)
			throws ServletException, IOException {
		response.setContentType("application/json");

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
				// TODO select the right one instead of forwarding to all?
				for(DianneInputs i : inputs){
					i.setInput(UUID.fromString(nnId), UUID.fromString(inputId), input);
				}
				
				ForwardListener inputListener = new ForwardListener(){
					@Override
					public void onForward(UUID moduleId, Tensor output, String... tags) {
						if(sses.get(input)!=null){
							JsonObject json = converter.toJson(output);
							StringBuilder builder = new StringBuilder();
							builder.append("data: ").append(json.toString()).append("\n\n");
							String msg = builder.toString();
						
							List<AsyncContext> list = sses.get(input);
							Iterator<AsyncContext> it = list.iterator();
							while(it.hasNext()){
								AsyncContext sse = it.next();
								try {
									PrintWriter writer = sse.getResponse().getWriter();
									writer.write(msg);
									writer.flush();
									if(writer.checkError()){
										System.err.println("Writer error: removing async endpoint.");
										writer.close();
										it.remove();									
									}
								} catch(Exception e){
									try { 
										sse.getResponse().getWriter().close(); 
									} catch (Exception ignore) {}
									it.remove();
									e.printStackTrace();
								}
							}
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
				ServiceRegistration<ForwardListener> r = context.registerService(ForwardListener.class, inputListener, properties);
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
			
			ServiceRegistration<ForwardListener> r = inputListeners.get(UUID.fromString(inputId));
			if(r!=null){
				r.unregister();
			}
		}
	}
}
