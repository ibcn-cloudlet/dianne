package be.iminds.iot.dianne.builder;

import java.io.IOException;

import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;

import be.iminds.iot.dianne.api.repository.DianneRepository;

import com.google.gson.JsonArray;
import com.google.gson.JsonPrimitive;

@Component(service = { javax.servlet.Servlet.class }, 
	property = { "alias:String=/dianne/load","aiolos.proxy=false" }, 
	immediate = true)
public class DianneLoader extends HttpServlet {
	
	private DianneRepository repository;
	
	@Reference
	public void setDianneRepository(DianneRepository repo){
		this.repository = repo;
	}
	
	@Override
	protected void doPost(HttpServletRequest request, HttpServletResponse response)
			throws ServletException, IOException {

		String action = request.getParameter("action");
		if("list".equals(action)){
			JsonArray names = new JsonArray();
			for(String name : repository.networks()){
				names.add(new JsonPrimitive(name));
			}
			response.getWriter().write(names.toString());
			response.getWriter().flush();
		} else if("load".equals(action)){
			String name = request.getParameter("name");
			
			response.getWriter().write("{\"modules\":");
			String modules = repository.loadNetwork(name);
			response.getWriter().write(modules);
			response.getWriter().write(", \"layout\":");
			String layout = repository.loadLayout(name);
			response.getWriter().write(layout);
			response.getWriter().write("}");
			response.getWriter().flush();
		}
	}
}
