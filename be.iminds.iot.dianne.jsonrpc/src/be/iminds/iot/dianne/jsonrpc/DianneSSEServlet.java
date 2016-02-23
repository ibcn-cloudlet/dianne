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
		data.add("message", new JsonPrimitive((String)event.getProperty("message")));
		data.add("level", new JsonPrimitive(event.getProperty("level").toString().toLowerCase()));
		long timestamp = (Long)event.getProperty("timestamp");
		data.add("time", new JsonPrimitive(timestamp));
		
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
