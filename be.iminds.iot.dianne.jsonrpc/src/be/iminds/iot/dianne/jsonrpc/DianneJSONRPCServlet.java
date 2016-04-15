package be.iminds.iot.dianne.jsonrpc;

import java.io.IOException;
import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;

import com.google.gson.stream.JsonReader;
import com.google.gson.stream.JsonWriter;

@Component(service = { javax.servlet.Servlet.class }, property = { "alias:String=/dianne/jsonrpc",
		"osgi.http.whiteboard.servlet.pattern=/dianne/jsonrpc", "aiolos.proxy=false" }, immediate = true)
public class DianneJSONRPCServlet extends HttpServlet {

	private JSONRPCRequestHandler handler;

	@Reference
	void setRequestHandler(JSONRPCRequestHandler h) {
		this.handler = h;
	}

	@Override
	protected void doPost(HttpServletRequest req, HttpServletResponse resp) throws ServletException, IOException {
		resp.setContentType("application/json");
		
		try (JsonReader reader = new JsonReader(req.getReader());
				JsonWriter writer = new JsonWriter(resp.getWriter())) {

			handler.handleRequest(reader, writer);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
