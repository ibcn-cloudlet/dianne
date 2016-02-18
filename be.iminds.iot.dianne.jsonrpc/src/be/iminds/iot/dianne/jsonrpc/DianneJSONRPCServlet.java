package be.iminds.iot.dianne.jsonrpc;

import java.io.IOException;
import java.io.InputStreamReader;

import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;

import com.google.gson.stream.JsonReader;
import com.google.gson.stream.JsonWriter;

@Component(service={javax.servlet.Servlet.class},
property={"alias:String=/dianne/jsonrpc",
	 	  "osgi.http.whiteboard.servlet.pattern=/dianne/jsonrpc",
		  "aiolos.proxy=false"},
immediate=true)
public class DianneJSONRPCServlet extends HttpServlet {

	private JSONRPCRequestHandler handler;

	@Override
	protected void doPost(HttpServletRequest req, HttpServletResponse resp) throws ServletException, IOException {
		try {
			JsonReader reader = new JsonReader(new InputStreamReader(req.getInputStream()));
			JsonWriter writer = new JsonWriter(resp.getWriter());
			
			resp.setContentType("application/json");
			handler.handleRequest(reader, writer);
		} catch(Exception e){
			e.printStackTrace();
		}
	}

	@Reference
	void setRequestHandler(JSONRPCRequestHandler h){
		this.handler = h;
	}
}
