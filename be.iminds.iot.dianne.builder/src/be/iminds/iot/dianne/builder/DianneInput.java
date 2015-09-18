package be.iminds.iot.dianne.builder;

import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Dictionary;
import java.util.HashMap;
import java.util.Hashtable;
import java.util.List;
import java.util.Map;
import java.util.UUID;

import javax.servlet.AsyncContext;
import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.osgi.framework.BundleContext;
import org.osgi.framework.ServiceRegistration;
import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;
import org.osgi.service.component.annotations.ReferenceCardinality;
import org.osgi.service.component.annotations.ReferencePolicy;

import be.iminds.iot.dianne.api.io.InputDescription;
import be.iminds.iot.dianne.api.io.InputManager;
import be.iminds.iot.dianne.api.nn.module.ForwardListener;
import be.iminds.iot.dianne.tensor.Tensor;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.google.gson.JsonPrimitive;

@Component(service = { javax.servlet.Servlet.class }, 
	property = { "alias:String=/dianne/input","aiolos.proxy=false" }, 
	immediate = true)
public class DianneInput extends HttpServlet {
	
	private List<InputManager> inputManagers = Collections.synchronizedList(new ArrayList<InputManager>());

	// Send event to UI when new input sample arrived
	private AsyncContext sse = null;
	private BundleContext context;
	private Map<UUID, ServiceRegistration> inputListeners = Collections.synchronizedMap(new HashMap<UUID, ServiceRegistration>());
	private JsonParser parser = new JsonParser();

	
	@Activate 
	public void activate(BundleContext context){
		this.context = context;
	}
	
	@Reference(cardinality=ReferenceCardinality.AT_LEAST_ONE, 
			policy=ReferencePolicy.DYNAMIC)
	public void addInputManager(InputManager mgr){
		this.inputManagers.add(mgr);
	}
	
	public void removeInputManager(InputManager mgr){
		this.inputManagers.remove(mgr);
	}
	
	@Override
	protected void doGet(HttpServletRequest request, HttpServletResponse response)
			throws ServletException, IOException {
		// write text/eventstream response
		response.setContentType("text/event-stream");
		response.setHeader("Cache-Control", "no-cache");
		response.setCharacterEncoding("UTF-8");
		response.addHeader("Connection", "keep-alive");
		
		sse = request.startAsync();
		sse.setTimeout(0); // no timeout => remove listener when error occurs.
	}
	
	@Override
	protected void doPost(HttpServletRequest request, HttpServletResponse response)
			throws ServletException, IOException {

		String action = request.getParameter("action");
		if("available-inputs".equals(action)){
			JsonArray inputs = new JsonArray();
			synchronized(inputManagers){
				for(InputManager m : inputManagers){
					for(InputDescription input : m.getAvailableInputs()){
						JsonObject o = new JsonObject();
						o.add("name", new JsonPrimitive(input.getName()));
						o.add("type", new JsonPrimitive(input.getType()));
						inputs.add(o);
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
					m.setInput(UUID.fromString(inputId), DianneDeployer.UI_NN_ID, input);
				}
				
				ForwardListener inputListener = new ForwardListener(){
					@Override
					public void onForward(UUID moduleId, Tensor output, String... tags) {
						if(sse!=null){
							try {
								JsonObject data = new JsonObject();

								if(output.dims().length==3){
									data.add("channels", new JsonPrimitive(output.dims()[0]));
									data.add("height", new JsonPrimitive(output.dims()[1]));
									data.add("width", new JsonPrimitive(output.dims()[2]));
								} else {
									data.add("channels", new JsonPrimitive(1));
									data.add("height", new JsonPrimitive(output.dims()[0]));
									data.add("width", new JsonPrimitive(output.dims()[1]));
								}
								if(tags!=null){
									JsonArray ta = new JsonArray();
									for(String t : tags){
										ta.add(new JsonPrimitive(t));
									}
									data.add("tags",ta);
								}
								
								data.add("data", parser.parse(Arrays.toString(output.get())));
								
								StringBuilder builder = new StringBuilder();
								builder.append("data: ").append(data.toString()).append("\n\n");
				
								PrintWriter writer = sse.getResponse().getWriter();
								writer.write(builder.toString());
								writer.flush();
							} catch(Exception e){}
						}
					}
				};
				Dictionary<String, Object> properties = new Hashtable<String, Object>();
				properties.put("targets", new String[]{DianneDeployer.UI_NN_ID+":"+inputId});
				properties.put("aiolos.unique", true);
				ServiceRegistration r = context.registerService(ForwardListener.class.getName(), inputListener, properties);
				inputListeners.put(UUID.fromString(inputId), r);
			}
		} else if("unsetinput".equals(action)){
			String inputId = request.getParameter("inputId");
			String input = request.getParameter("input");
			// TODO only forward to applicable inputmgr?
			synchronized(inputManagers){
				for(InputManager m : inputManagers){
					m.unsetInput(UUID.fromString(inputId), DianneDeployer.UI_NN_ID, input);
				}
			}
			
			ServiceRegistration r = inputListeners.get(UUID.fromString(inputId));
			if(r!=null){
				r.unregister();
			}
		}
	}
}
