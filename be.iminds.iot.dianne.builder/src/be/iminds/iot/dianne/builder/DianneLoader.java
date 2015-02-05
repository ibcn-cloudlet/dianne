package be.iminds.iot.dianne.builder;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;
import org.osgi.service.http.HttpService;

@Component(service = { javax.servlet.Servlet.class }, 
	property = { "alias:String=/dianne/load","aiolos.proxy=false" }, 
	immediate = true)
public class DianneLoader extends HttpServlet {
	
	@Reference
	public void setHttpService(HttpService http){
		try {
			// Use manual registration - problems with whiteboard
			http.registerServlet("/dianne/load", this, null, null);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	@Override
	protected void doPost(HttpServletRequest request, HttpServletResponse response)
			throws ServletException, IOException {

		String modules = new String(Files.readAllBytes(Paths.get("nn/modules.txt")));
		String layout = new String(Files.readAllBytes(Paths.get("nn/layout.txt")));
		
		String result = "{\"modules\":"+modules+", \"layout\":"+layout+"}";
		response.getWriter().write(result);
		response.getWriter().flush();
	}
	

}
