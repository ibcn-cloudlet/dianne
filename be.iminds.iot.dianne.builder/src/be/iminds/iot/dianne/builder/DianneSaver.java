package be.iminds.iot.dianne.builder;

import java.io.IOException;

import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;

import be.iminds.iot.dianne.api.repository.DianneRepository;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonParser;

@Component(service = { javax.servlet.Servlet.class }, 
	property = { "alias:String=/dianne/save","aiolos.proxy=false" }, 
	immediate = true)
public class DianneSaver extends HttpServlet {

	private DianneRepository repository;
	
	@Reference
	public void setDianneRepository(DianneRepository repo){
		this.repository = repo;
	}
	
	@Override
	protected void doPost(HttpServletRequest request, HttpServletResponse response)
			throws ServletException, IOException {
		String name = request.getParameter("name");
		String modules = request.getParameter("modules");
		String layout = request.getParameter("layout");
		
		// Parse and rewrite to get pretty output format
		JsonParser parser = new JsonParser();
		Gson gson = new GsonBuilder().setPrettyPrinting().create();
		String jsonOutput = gson.toJson(parser.parse(modules));
		
		// TODO which formats to save?
		repository.storeNeuralNetwork(name, modules);
		repository.storeLayout(name, layout);
		
		response.getWriter().println("{}");
		response.getWriter().flush();
	}
	

}
