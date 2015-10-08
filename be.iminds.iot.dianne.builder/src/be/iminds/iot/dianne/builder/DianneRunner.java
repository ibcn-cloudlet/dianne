package be.iminds.iot.dianne.builder;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.io.PrintWriter;
import java.net.URL;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Dictionary;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.Hashtable;
import java.util.Map;
import java.util.Random;
import java.util.UUID;
import java.util.stream.Collectors;

import javax.imageio.ImageIO;
import javax.servlet.AsyncContext;
import javax.servlet.AsyncEvent;
import javax.servlet.AsyncListener;
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
import be.iminds.iot.dianne.api.nn.Dianne;
import be.iminds.iot.dianne.api.nn.NeuralNetwork;
import be.iminds.iot.dianne.api.nn.module.ForwardListener;
import be.iminds.iot.dianne.api.nn.module.Module;
import be.iminds.iot.dianne.api.nn.module.Module.Mode;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkInstanceDTO;
import be.iminds.iot.dianne.api.nn.platform.DiannePlatform;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;
import be.iminds.iot.dianne.tensor.util.ImageConverter;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.google.gson.JsonPrimitive;

@Component(service = { javax.servlet.Servlet.class }, 
	property = { "alias:String=/dianne/run",
		 		 "osgi.http.whiteboard.servlet.pattern=/dianne/run",
		 		 "osgi.http.whiteboard.servlet.asyncSupported:Boolean=true",
				 "aiolos.proxy=false" }, 
	immediate = true)
public class DianneRunner extends HttpServlet {
	
	private BundleContext context;
	
	private TensorFactory factory;
	private ImageConverter converter;
	
	private JsonParser parser = new JsonParser();
	
	// also keep datasets to already forward random sample while sending sample to the ui
	private Random rand = new Random(System.currentTimeMillis());
	private Map<String, Dataset> datasets = Collections.synchronizedMap(new HashMap<String, Dataset>());
	private Dianne dianne;
	private DiannePlatform platform;
	
	@Activate
	public void activate(BundleContext c){
		this.context = c;
	}
	
	@Reference
	void setTensorFactory(TensorFactory factory){
		this.factory = factory;
		this.converter = new ImageConverter(factory);
	}
	
	@Reference
	void setDianne(Dianne d){
		dianne = d;
	}
	
	@Reference
	void setDiannePlatform(DiannePlatform p){
		platform = p;
	}
	
	@Reference(cardinality=ReferenceCardinality.MULTIPLE, 
			policy=ReferencePolicy.DYNAMIC)
	void addDataset(Dataset dataset, Map<String, Object> properties){
		String name = (String) properties.get("name");
		this.datasets.put(name, dataset);
	}
	
	void removeDataset(Dataset dataset, Map<String, Object> properties){
		String name = (String) properties.get("name");
		datasets.remove(name);
	}
	
	@Override
	protected void doGet(HttpServletRequest request, HttpServletResponse response)
			throws ServletException, IOException {
		// write text/eventstream response
		response.setContentType("text/event-stream");
		response.setHeader("Cache-Control", "no-cache");
		response.setCharacterEncoding("UTF-8");
		response.addHeader("Connection", "keep-alive");
		
		// register forward listener for this
		String nnId = request.getParameter("nnId");
		if(nnId == null){
			return;
		}
		
		NeuralNetworkInstanceDTO nn = platform.getNeuralNetworkInstance(UUID.fromString(nnId));
		if(nn!=null){
			
			try {
				Map<UUID, String[]> labels = nn.modules.values().stream().map(i -> i.module)
				.filter(m -> m.type.equals("Output"))
				.filter(m -> m.properties.containsKey("labels"))
				.collect(Collectors.toMap(m -> m.id, m -> {
					String l = m.properties.get("labels");
					l = l.substring(1, l.length()-1);
					return l.split(",");
				}));
				SSEForwardListener listener = new SSEForwardListener(nnId, labels, request.startAsync());
				listener.register(context);
			} catch(Exception e){
				e.printStackTrace();
			}
		}
	}
	
	@Override
	protected void doPost(HttpServletRequest request, HttpServletResponse response)
			throws ServletException, IOException {
		String id = request.getParameter("id");
		if(id == null){
			System.out.println("No neural network instance specified");
			return;
		}
		UUID nnId = UUID.fromString(id);
		NeuralNetworkInstanceDTO nni = platform.getNeuralNetworkInstance(nnId);
		if(nni==null){
			System.out.println("Neural network instance "+id+" not deployed");
			return;
		}
		
		NeuralNetwork nn = dianne.getNeuralNetwork(nni);
		if(nn==null){
			System.out.println("Neural network instance "+id+" not available");
			return;
		}
		
		
		if(request.getParameter("forward")!=null){
			String inputId = request.getParameter("input");
			
			JsonObject sample = parser.parse(request.getParameter("forward")).getAsJsonObject();
			int channels = sample.get("channels").getAsInt();
			int width = sample.get("width").getAsInt();
			int height = sample.get("height").getAsInt();

			float[] data = parseInput(sample.get("data").getAsJsonArray().toString());
			Tensor t = factory.createTensor(data, channels, height, width);
			
			nn.aforward(UUID.fromString(inputId), t);
			
		} else if(request.getParameter("url")!=null){
			String url = request.getParameter("url");
			String inputId = request.getParameter("input");
			
			Tensor t = null;
			try {
				URL u = new URL(url);
				BufferedImage img = ImageIO.read(u);
				t = converter.readFromImage(img);
			} catch(Exception e){
				System.out.println("Failed to read image from url "+url);
				return;
			}
			
			nn.aforward(UUID.fromString(inputId), t);
			
		} else if(request.getParameter("mode")!=null){
			String mode = request.getParameter("mode");
			String targetId = request.getParameter("target");

			Module m = nn.getModules().get(UUID.fromString(targetId));
			if(m!=null){
				m.setMode(EnumSet.of(Mode.valueOf(mode)));
			}
		} else if(request.getParameter("dataset")!=null){
			String dataset = request.getParameter("dataset");
			Dataset d = datasets.get(dataset);
			
			if(d!=null){
				String inputId = request.getParameter("input");

				Tensor t = d.getInputSample(rand.nextInt(d.size()));
				
				nn.aforward(UUID.fromString(inputId), t);
				
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
	
	private String outputSSEMessage(UUID outputId, String[] outputLabels, Tensor output, String...tags){
		JsonObject data = new JsonObject();

		// format output as [['label', val],['label2',val2],...] for in highcharts
		String[] labels;
		float[] values;
		if(output.size()>10){
			// if more than 10 outputs, only send top-10 results
			Integer[] indices = new Integer[output.size()];
			for(int i=0;i<output.size();i++){
				indices[i] = i;
			}
			Arrays.sort(indices, new Comparator<Integer>() {
				@Override
				public int compare(Integer o1, Integer o2) {
					float v1 = output.get(o1);
					float v2 = output.get(o2);
					// inverse order to have large->small order
					return v1 > v2 ? -1 : (v1 < v2 ? 1 : 0);
				}
			});
			labels = new String[10];
			values = new float[10];
			for(int i=0;i<10;i++){
				labels[i] = outputLabels!=null ? outputLabels[indices[i]] : ""+indices[i];
				values[i] = output.get(indices[i]);
			}
		} else {
			labels = outputLabels;
			if(labels==null){
				labels = new String[output.size()];
				for(int i=0;i<labels.length;i++){
					labels[i] = ""+i;
				}
			}
			values = output.get();
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
		
		data.add("id", new JsonPrimitive(outputId.toString()));
		
		StringBuilder builder = new StringBuilder();
		builder.append("data: ").append(data.toString()).append("\n\n");
		return builder.toString();
	}
	
	private class SSEForwardListener implements ForwardListener {

		private final String nnId;
		private final Map<UUID, String[]> labels;
		private final AsyncContext async;
		private ServiceRegistration reg;
		
		public SSEForwardListener(String nnId,
				Map<UUID, String[]> labels,
				AsyncContext async) {
			this.nnId = nnId;
			this.labels = labels;
			this.async = async;
			this.async.setTimeout(300000); // let it ultimately timeout if client is closed
			
			this.async.addListener(new AsyncListener() {
				@Override
				public void onTimeout(AsyncEvent e) throws IOException {
					unregister();
				}
				@Override
				public void onStartAsync(AsyncEvent e) throws IOException {
					async.getResponse().getWriter().println("ping");
					if(async.getResponse().getWriter().checkError()){
						async.complete();
					}
				}
				@Override
				public void onError(AsyncEvent e) throws IOException {
					unregister();
				}
				@Override
				public void onComplete(AsyncEvent e) throws IOException {
					unregister();
				}
			});
		}
		
		public void register(BundleContext context){
			Dictionary<String, Object> props = new Hashtable();
			props.put("targets", new String[]{nnId});
			props.put("aiolos.unique", true);
			reg = context.registerService(ForwardListener.class, this, props);
		}
		
		public void unregister(){
			reg.unregister();
		}
		
		@Override
		public void onForward(UUID moduleId, Tensor output, String... tags) {
			try {
				String sseMessage = outputSSEMessage(moduleId, labels.get(moduleId), output, tags);
				PrintWriter writer = async.getResponse().getWriter();
				writer.write(sseMessage);
				writer.flush();
				if(writer.checkError()){
					unregister();
				}
			} catch(Exception e){
				e.printStackTrace();
				unregister();
			}	
		}
	}
}
