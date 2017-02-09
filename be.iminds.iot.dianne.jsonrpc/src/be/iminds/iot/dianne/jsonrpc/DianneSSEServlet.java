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
package be.iminds.iot.dianne.jsonrpc;

import java.io.IOException;
import java.io.PrintWriter;
import java.util.Iterator;
import java.util.Map;
import java.util.Map.Entry;
import java.util.concurrent.ConcurrentHashMap;

import javax.servlet.AsyncContext;
import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.osgi.service.component.annotations.Component;
import org.osgi.service.event.Event;
import org.osgi.service.event.EventHandler;

import com.google.gson.JsonObject;
import com.google.gson.JsonPrimitive;

@Component(service={javax.servlet.Servlet.class, EventHandler.class},
property={"alias:String=/dianne/sse",
	 	  "osgi.http.whiteboard.servlet.pattern=/dianne/sse",
		  "aiolos.proxy=false",
		  "event.topics=dianne/*"},
immediate=true)
public class DianneSSEServlet extends HttpServlet implements EventHandler {

	private static final long serialVersionUID = 1L;
	
	private Map<String, AsyncContext> clients = new ConcurrentHashMap<>();
	
	@Override
	protected void doGet(HttpServletRequest request, HttpServletResponse response)
			throws ServletException, IOException {
		// write text/eventstream response
		response.setContentType("text/event-stream");
		response.setHeader("Cache-Control", "no-cache");
		response.setCharacterEncoding("UTF-8");
		response.addHeader("Connection", "keep-alive");
		
		String client = request.getRemoteHost()+":"+request.getRemotePort();
		AsyncContext async = request.startAsync();
		async.setTimeout(30000);
		clients.put(client, async);
	}

	@Override
	public void handleEvent(Event event) {
		// construct server sent event
		JsonObject data = new JsonObject();
		
		if(event.getTopic().contains("progress")){
			// progress
			data.add("type", new JsonPrimitive("progress"));
			data.add("jobId", new JsonPrimitive(event.getProperty("jobId").toString()));

			if(event.containsProperty("iteration")){
				data.add("iteration", new JsonPrimitive((Long)event.getProperty("iteration")));
			}
			
			if(event.containsProperty("minibatchLoss")){
				data.add("minibatchLoss", new JsonPrimitive((Float)event.getProperty("minibatchLoss")));
			}
			
			if(event.containsProperty("validationLoss")){
				data.add("validationLoss", new JsonPrimitive((Float)event.getProperty("validationLoss")));
			}
			
			if(event.containsProperty("q")){
				data.add("q", new JsonPrimitive((Float)event.getProperty("q")));
			}
			
			if(event.containsProperty("reward")){
				data.add("reward", new JsonPrimitive((Float)event.getProperty("reward")));
			}
			
			if(event.containsProperty("sequence")){
				data.add("sequence", new JsonPrimitive((Long)event.getProperty("sequence")));
			}
			
			if(event.containsProperty("worker")){
				data.add("worker", new JsonPrimitive((Integer)event.getProperty("worker")));
			}
		} else {
			// notification
			if(event.containsProperty("jobId"))
				data.add("jobId", new JsonPrimitive(event.getProperty("jobId").toString()));
			data.add("type", new JsonPrimitive("notification"));
			data.add("message", new JsonPrimitive((String)event.getProperty("message")));
			data.add("level", new JsonPrimitive(event.getProperty("level").toString()));
			long timestamp = (Long)event.getProperty("timestamp");
			data.add("timestamp", new JsonPrimitive(timestamp));
		}
		
		StringBuilder builder = new StringBuilder();
		builder.append("data: ").append(data.toString()).append("\n\n");
		String sse = builder.toString();
		
		// send to all clients
		Iterator<Entry<String,AsyncContext>> it = clients.entrySet().iterator();
		while(it.hasNext()){
			AsyncContext client = it.next().getValue();
			try {
				PrintWriter writer = client.getResponse().getWriter();
				writer.write(sse);
				writer.flush();
			} catch(Exception e){
				it.remove();
			}
		}
	}
	
}
