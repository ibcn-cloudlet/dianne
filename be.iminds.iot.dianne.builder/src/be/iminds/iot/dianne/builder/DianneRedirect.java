package be.iminds.iot.dianne.builder;

import java.io.IOException;

import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.osgi.service.component.annotations.Component;

@Component(service={javax.servlet.Servlet.class},
	property={"alias:String=/dianne",
		 	  "osgi.http.whiteboard.servlet.pattern=/dianne",
			  "aiolos.proxy=false"},
	immediate=true)
public class DianneRedirect extends HttpServlet {

	@Override
	protected void doGet(HttpServletRequest req, HttpServletResponse resp) throws ServletException, IOException {
		resp.sendRedirect("dianne/ui/dianne.html");
	}

}
