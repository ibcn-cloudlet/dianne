package be.iminds.iot.dianne.rl.pong.ui;

import javax.servlet.http.HttpServlet;

import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;
import org.osgi.service.http.HttpService;

@Component(service={javax.servlet.Servlet.class},
	property={"alias:String=/pong","aiolos.proxy=false"},
	immediate=true)
public class PongServlet extends HttpServlet {

	
	@Reference
	public void setHttpService(HttpService http){
		try {
			// TODO How to register resources with whiteboard pattern?
			http.registerResources("/pong", "res", null);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
