package be.iminds.iot.dianne.dashboard;

import java.io.IOException;

import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;
import org.osgi.service.http.HttpService;

@Component(service={javax.servlet.Servlet.class},
property={"alias:String=/dianne/dashboard",
	 	  "osgi.http.whiteboard.servlet.pattern=/dianne/dashboard",
		  "aiolos.proxy=false"},
immediate=true)
public class DianneDashboard extends HttpServlet{

	@Reference
	void setHttpService(HttpService http){
		try {
			// TODO How to register resources with whiteboard pattern?
			http.registerResources("/dianne/ui/dashboard", "res", null);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	@Override
	protected void doGet(HttpServletRequest req, HttpServletResponse resp) throws ServletException, IOException {
		resp.sendRedirect("/dianne/ui/dashboard/dashboard.html");
	}
}
