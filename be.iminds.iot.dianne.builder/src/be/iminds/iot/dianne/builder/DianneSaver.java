package be.iminds.iot.dianne.builder;

import java.io.IOException;

import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;

import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkDTO;
import be.iminds.iot.dianne.api.repository.DianneRepository;
import be.iminds.iot.dianne.nn.util.DianneJSONConverter;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonParser;

@Component(service = { javax.servlet.Servlet.class }, 
	property = { "alias:String=/dianne/save",
		 		 "osgi.http.whiteboard.servlet.pattern=/dianne/save",
				 "aiolos.proxy=false" }, 
	immediate = true)
public class DianneSaver extends HttpServlet {

	private DianneRepository repository;
	
	@Reference
	void setDianneRepository(DianneRepository repo){
		this.repository = repo;
	}
	
	@Override
	protected void doPost(HttpServletRequest request, HttpServletResponse response)
			throws ServletException, IOException {
		String nn = request.getParameter("nn");
		String layout = request.getParameter("layout");
		
		NeuralNetworkDTO dto = DianneJSONConverter.parseJSON(nn);
		repository.storeNeuralNetwork(dto);
		repository.storeLayout(dto.name, layout);
		
		response.getWriter().println("{}");
		response.getWriter().flush();
	}
	

}
