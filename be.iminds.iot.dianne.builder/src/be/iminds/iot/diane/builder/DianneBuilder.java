package be.iminds.iot.diane.builder;

import java.io.IOException;

import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;
import org.osgi.service.http.HttpService;

@Component(service={javax.servlet.Servlet.class},
	property={"alias:String=/dianne/builder"},
	immediate=true)
public class DianneBuilder extends HttpServlet {

	@Reference
	public void setHttpService(HttpService http){
		try {
			// Use manual registration - problems with whiteboard
			http.registerResources("/dianne", "res", null);
			http.registerServlet("/dianne/builder", this, null, null);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	
	@Override
	protected void doGet(HttpServletRequest request, HttpServletResponse response)
			throws ServletException, IOException {
	}
	
	@Override
	protected void doPost(HttpServletRequest request, HttpServletResponse response)
			throws ServletException, IOException {

	}
}
