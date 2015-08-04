package be.iminds.iot.dianne.builder;

import java.io.IOException;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Dictionary;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.Hashtable;
import java.util.Iterator;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;

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

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.nn.module.ForwardListener;
import be.iminds.iot.dianne.api.nn.module.Input;
import be.iminds.iot.dianne.api.nn.module.Module;
import be.iminds.iot.dianne.api.nn.module.Module.Mode;
import be.iminds.iot.dianne.api.nn.module.Output;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.google.gson.JsonPrimitive;

@Component(service = { javax.servlet.Servlet.class }, 
	property = { "alias:String=/dianne/run","aiolos.proxy=false" }, 
	immediate = true)
public class DianneRunner extends HttpServlet {
	
	private BundleContext context;
	
	private TensorFactory factory;
	
	private JsonParser parser = new JsonParser();
	
	// for now fixed 1 input, 1 output, and trainable modules
	private Map<String, Module> modules = new HashMap<String, Module>();

	// also keep datasets to already forward random sample while sending sample to the ui
	private Random rand = new Random(System.currentTimeMillis());
	private Map<String, Dataset> datasets = Collections.synchronizedMap(new HashMap<String, Dataset>());

	// register a forwardlistener for each output?
	private Map<Module, ServiceRegistration> forwardListeners = Collections.synchronizedMap(new HashMap<Module, ServiceRegistration>());
	
	private long t1,t2;
	
	private AsyncContext sse = null;
	
	@Activate
	public void activate(BundleContext c){
		this.context = c;
	}
	
	@Reference
	public void setTensorFactory(TensorFactory factory){
		this.factory = factory;
	}
	
	@Reference(cardinality=ReferenceCardinality.MULTIPLE, 
			policy=ReferencePolicy.DYNAMIC)
	public void addDataset(Dataset dataset, Map<String, Object> properties){
		String name = (String) properties.get("name");
		this.datasets.put(name, dataset);
	}
	
	public void removeDataset(Dataset dataset, Map<String, Object> properties){
		String name = (String) properties.get("name");
		datasets.remove(name);
	}
	
	@Reference(cardinality=ReferenceCardinality.MULTIPLE, 
			policy=ReferencePolicy.DYNAMIC)
	public void addModule(Module m){
		this.modules.put(m.getId().toString(), m);
		if(m instanceof Output){
			final Output output = (Output) m;
			final String id  = m.getId().toString();
			
			Dictionary<String, Object> properties = new Hashtable<String, Object>();
			properties.put("targets", new String[]{DianneDeployer.UI_NN_ID+":"+id});
			properties.put("aiolos.unique", true);
		
			ForwardListener listener = new ForwardListener() {
				@Override
				public void onForward(Tensor t, String... tags) {
					if(sse!=null){
						t2 = System.currentTimeMillis();
						System.out.println("FORWARD TIME "+(t2-t1)+" ms.");
						try {
							JsonObject data = new JsonObject();

							// format output as [['label', val],['label2',val2],...] for in highcharts
							String[] labels;
							float[] values;
							if(t.size()>10){
								// if more than 10 outputs, only send top-10 results
								Integer[] indices = new Integer[t.size()];
								for(int i=0;i<t.size();i++){
									indices[i] = i;
								}
								Arrays.sort(indices, new Comparator<Integer>() {
									@Override
									public int compare(Integer o1, Integer o2) {
										float v1 = t.get(o1);
										float v2 = t.get(o2);
										// inverse order to have large->small order
										return v1 > v2 ? -1 : (v1 < v2 ? 1 : 0);
									}
								});
								labels = new String[10];
								values = new float[10];
								for(int i=0;i<10;i++){
									labels[i] = output.getOutputLabels()!=null ? output.getOutputLabels()[indices[i]] : ""+indices[i];
									values[i] = t.get(indices[i]);
								}
							} else {
								labels = output.getOutputLabels();
								values = t.get();
							}
							
							JsonArray result = new JsonArray();
							for(int i=0;i<values.length;i++){
								result.add(new JsonPrimitive(values[i]));
							}
							data.add("output", result);
							
							JsonArray l = new JsonArray();
							for(int i=0;i<labels.length;i++){
								l.add(new JsonPrimitive(labels[i]));
							}
							data.add("labels", l);
							
							if(tags!=null){
								JsonArray ta = new JsonArray();
								for(String tt : tags){
									ta.add(new JsonPrimitive(tt));
								}
								data.add("tags",ta);
							}
							
							data.add("id", new JsonPrimitive(id));
							
							StringBuilder builder = new StringBuilder();
							builder.append("data: ").append(data.toString()).append("\n\n");
			
							PrintWriter writer = sse.getResponse().getWriter();
							writer.write(builder.toString());
							writer.flush();
							if(writer.checkError()){
								sse = null;
							}
						} catch(Exception e){
							e.printStackTrace();
							sse = null;
						}
					}
				}
			};

			ServiceRegistration r = context.registerService(ForwardListener.class.getName(), listener, properties);
			
			forwardListeners.put(output, r);
		} else if(m instanceof Input){
			final Input input = (Input) m;
			final String id  = m.getId().toString();
			
			Dictionary<String, Object> properties = new Hashtable<String, Object>();
			properties.put("targets", new String[]{DianneDeployer.UI_NN_ID+":"+id});
			properties.put("aiolos.unique", true);
		
			ForwardListener listener = new ForwardListener() {
				@Override
				public void onForward(Tensor t, String... tags) {
					t1 = System.currentTimeMillis();
				}
			};
			ServiceRegistration r = context.registerService(ForwardListener.class.getName(), listener, properties);
			forwardListeners.put(input, r);
		}
	}
	
	public void removeModule(Module m){
		Iterator<Entry<String, Module>> it = modules.entrySet().iterator();
		while(it.hasNext()){
			Entry<String, Module> e = it.next();
			if(e.getValue()==m){
				it.remove();
			}
		}
		
		ServiceRegistration r = forwardListeners.get(m);
		if(r!=null){
			r.unregister();
		}
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
		if(request.getParameter("forward")!=null){
			String inputId = request.getParameter("input");
			Input input = (Input) modules.get(inputId);
			
			JsonObject sample = parser.parse(request.getParameter("forward")).getAsJsonObject();
			int channels = sample.get("channels").getAsInt();
			int width = sample.get("width").getAsInt();
			int height = sample.get("height").getAsInt();

			float[] data = parseInput(sample.get("data").getAsJsonArray().toString());
			Tensor t = factory.createTensor(data, channels, height, width);
			input.input(t);
		} else if(request.getParameter("mode")!=null){
			String mode = request.getParameter("mode");
			String targetId = request.getParameter("target");

			Module m = modules.get(targetId);
			if(m!=null){
				m.setMode(EnumSet.of(Mode.valueOf(mode)));
			}
		} else if(request.getParameter("dataset")!=null){
			String dataset = request.getParameter("dataset");
			Dataset d = datasets.get(dataset);
			
			if(d!=null){
				String inputId = request.getParameter("input");
				Input input = (Input) modules.get(inputId);
				
				Tensor t = d.getInputSample(rand.nextInt(d.size()));
				if(input!=null){
					input.input(t);
				}
				
				JsonObject sample = new JsonObject();
				if(t.dims().length==3){
					sample.add("channels", new JsonPrimitive(t.dims()[0]));
					sample.add("height", new JsonPrimitive(t.dims()[1]));
					sample.add("width", new JsonPrimitive(t.dims()[2]));
				} else {
					sample.add("channels", new JsonPrimitive(1));
					sample.add("height", new JsonPrimitive(t.dims()[0]));
					sample.add("width", new JsonPrimitive(t.dims()[1]));
				}
				sample.add("data", parser.parse(Arrays.toString(t.get())));
				response.getWriter().println(sample.toString());
				response.getWriter().flush();
			}
		} 
	}
	
	private float[] parseInput(String string){
		String[] strings = string.replace("[", "").replace("]", "").split(",");
		float result[] = new float[strings.length];
		for (int i = 0; i < result.length; i++) {
			result[i] = Float.parseFloat(strings[i]);
		}
		return result;
	}
}
