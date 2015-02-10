package be.iminds.iot.dianne.builder;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.osgi.framework.BundleContext;
import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Component;

import com.google.gson.JsonArray;
import com.google.gson.JsonPrimitive;

@Component(service = { javax.servlet.Servlet.class }, 
	property = { "alias:String=/dianne/load","aiolos.proxy=false" }, 
	immediate = true)
public class DianneLoader extends HttpServlet {
	
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

		String action = request.getParameter("action");
		if("list".equals(action)){
			JsonArray names = new JsonArray();
			File dir = new File(storage);
			for(File f : dir.listFiles()){
				if(f.isDirectory()){
					names.add(new JsonPrimitive(f.getName()));
				}
			}
			response.getWriter().write(names.toString());
			response.getWriter().flush();
		} else if("load".equals(action)){
			String name = request.getParameter("name");
			String modules = new String(Files.readAllBytes(Paths.get(storage+"/"+name+"/modules.txt")));
			String layout = new String(Files.readAllBytes(Paths.get(storage+"/"+name+"/layout.txt")));
			
			String result = "{\"modules\":"+modules+", \"layout\":"+layout+"}";
			response.getWriter().write(result);
			response.getWriter().flush();
		}
		

	}
	

}
