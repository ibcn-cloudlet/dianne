package be.iminds.iot.dianne.builder;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;

import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.osgi.framework.BundleContext;
import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Component;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonParser;

@Component(service = { javax.servlet.Servlet.class }, 
	property = { "alias:String=/dianne/save","aiolos.proxy=false" }, 
	immediate = true)
public class DianneSaver extends HttpServlet {

	private String storage = "nn";
	
	@Activate
	public void activate(BundleContext context){
		String s = context.getProperty("be.iminds.iot.dianne.storage");
		if(s!=null){
			storage = s;
		}
	}
	
	@Override
	protected void doPost(HttpServletRequest request, HttpServletResponse response)
			throws ServletException, IOException {
		String name = request.getParameter("name");
		String modules = request.getParameter("modules");
		String layout = request.getParameter("layout");
		
		File dir = new File(storage+"/"+name);
		if(!dir.exists()){
			dir.mkdirs();
		}
		
		// Parse and rewrite to get pretty output format
		JsonParser parser = new JsonParser();
		Gson gson = new GsonBuilder().setPrettyPrinting().create();
		String jsonOutput = gson.toJson(parser.parse(modules));
		
		// TODO which formats to save?
		File m = new File(storage+"/"+name+"/modules.txt");
		PrintWriter p = new PrintWriter(m);
		p.write(jsonOutput);
		p.close();
		
		File l = new File(storage+"/"+name+"/layout.txt");
		PrintWriter p2 = new PrintWriter(l);
		p2.write(layout);
		p2.close();		
		
		response.getWriter().println("{}");
		response.getWriter().flush();
	}
	

}
