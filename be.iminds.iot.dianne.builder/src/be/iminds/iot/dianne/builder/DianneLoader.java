package be.iminds.iot.dianne.builder;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.osgi.service.component.annotations.Component;

@Component(service = { javax.servlet.Servlet.class }, 
	property = { "alias:String=/dianne/load","aiolos.proxy=false" }, 
	immediate = true)
public class DianneLoader extends HttpServlet {
	
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
