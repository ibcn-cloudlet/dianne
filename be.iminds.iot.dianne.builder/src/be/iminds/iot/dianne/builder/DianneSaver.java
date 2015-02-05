package be.iminds.iot.dianne.builder;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintWriter;

import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;
import org.osgi.service.http.HttpService;

@Component(service = { javax.servlet.Servlet.class }, 
	property = { "alias:String=/dianne/save","aiolos.proxy=false" }, 
	immediate = true)
public class DianneSaver extends HttpServlet {
	
	@Reference
	public void setHttpService(HttpService http){
		try {
			// Use manual registration - problems with whiteboard
			http.registerServlet("/dianne/save", this, null, null);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	@Override
	protected void doPost(HttpServletRequest request, HttpServletResponse response)
			throws ServletException, IOException {
		String modules = request.getParameter("modules");
		String layout = request.getParameter("layout");
		
		File dir = new File("nn");
		if(!dir.exists()){
			dir.mkdirs();
		}
		
		// TODO which formats to save?
		File m = new File("nn/modules.txt");
		PrintWriter p = new PrintWriter(m);
		p.write(modules);
		p.close();
		
		File l = new File("nn/layout.txt");
		PrintWriter p2 = new PrintWriter(l);
		p2.write(layout);
		p2.close();		
		
		response.getWriter().println("{'result':'succes'}");
		response.getWriter().flush();
	}
	

}
