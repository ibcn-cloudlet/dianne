package be.iminds.iot.dianne.builder;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.UUID;

import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;
import org.osgi.service.component.annotations.ReferenceCardinality;
import org.osgi.service.component.annotations.ReferencePolicy;

import be.iminds.iot.dianne.api.io.InputManager;

import com.google.gson.JsonArray;
import com.google.gson.JsonPrimitive;

@Component(service = { javax.servlet.Servlet.class }, 
	property = { "alias:String=/dianne/input","aiolos.proxy=false" }, 
	immediate = true)
public class DianneInput extends HttpServlet {
	
	private List<InputManager> inputManagers = Collections.synchronizedList(new ArrayList<InputManager>());
	
	@Reference(cardinality=ReferenceCardinality.MULTIPLE, 
			policy=ReferencePolicy.DYNAMIC)
	public void addInputManager(InputManager mgr){
		this.inputManagers.add(mgr);
	}
	
	public void removeInputManager(InputManager mgr){
		this.inputManagers.remove(mgr);
	}
	
	@Override
	protected void doPost(HttpServletRequest request, HttpServletResponse response)
			throws ServletException, IOException {

		String action = request.getParameter("action");
		if("available-inputs".equals(action)){
			JsonArray inputs = new JsonArray();
			synchronized(inputManagers){
				for(InputManager m : inputManagers){
					for(String input : m.getAvailableInputs()){
						inputs.add(new JsonPrimitive(input));
					}
				}
			}
			response.getWriter().write(inputs.toString());
			response.getWriter().flush();
		} else if("setinput".equals(action)){
			String inputId = request.getParameter("inputId");
			String input = request.getParameter("input");
			// TODO only forward to applicable inputmgr?
			synchronized(inputManagers){
				for(InputManager m : inputManagers){
					m.setInput(UUID.fromString(inputId), input);
				}
			}
		} else if("unsetinput".equals(action)){
			String inputId = request.getParameter("inputId");
			String input = request.getParameter("input");
			// TODO only forward to applicable inputmgr?
			synchronized(inputManagers){
				for(InputManager m : inputManagers){
					m.unsetInput(UUID.fromString(inputId), input);
				}
			}
		}
	}
}
