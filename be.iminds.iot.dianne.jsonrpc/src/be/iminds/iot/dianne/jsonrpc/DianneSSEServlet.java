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
		
		for(String key : event.getPropertyNames()){
			if(!key.equals("event.topics"))
				data.add(key, new JsonPrimitive(event.getProperty(key).toString()));
		}
		
		if(event.getTopic().contains("progress")){
			// progress
			data.add("type", new JsonPrimitive("progress"));
		} else {
			data.add("type", new JsonPrimitive("notification"));
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
