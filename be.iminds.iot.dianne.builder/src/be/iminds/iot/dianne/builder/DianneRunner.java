/*******************************************************************************
 * DIANNE  - Framework for distributed artificial neural networks
 * Copyright (C) 2015  iMinds - IBCN - UGent
 *
 * This file is part of DIANNE.
 *
 * DIANNE is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * Contributors:
 *     Tim Verbelen, Steven Bohez
 *******************************************************************************/
package be.iminds.iot.dianne.builder;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.io.PrintWriter;
import java.net.URL;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Dictionary;
import java.util.EnumSet;
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

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.google.gson.JsonPrimitive;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.dataset.DianneDatasets;
import be.iminds.iot.dianne.api.dataset.Sample;
import be.iminds.iot.dianne.api.nn.Dianne;
import be.iminds.iot.dianne.api.nn.NeuralNetwork;
import be.iminds.iot.dianne.api.nn.module.ForwardListener;
import be.iminds.iot.dianne.api.nn.module.Module;
import be.iminds.iot.dianne.api.nn.module.Module.Mode;
import be.iminds.iot.dianne.api.nn.module.ModuleException;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkInstanceDTO;
import be.iminds.iot.dianne.api.nn.platform.DiannePlatform;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;
import be.iminds.iot.dianne.tensor.util.ImageConverter;
import be.iminds.iot.dianne.tensor.util.JsonConverter;

@Component(service = { javax.servlet.Servlet.class }, 
	property = { "alias:String=/dianne/run",
		 		 "osgi.http.whiteboard.servlet.pattern=/dianne/run",
		 		 "osgi.http.whiteboard.servlet.asyncSupported:Boolean=true",
				 "aiolos.proxy=false" }, 
	immediate = true)
public class DianneRunner extends HttpServlet {
	
	private static final long serialVersionUID = 1L;

	private BundleContext context;
	
	private ImageConverter imageConverter;
	
	private JsonParser parser = new JsonParser();
	private JsonConverter jsonConverter = new JsonConverter();
	
	// also keep datasets to already forward random sample while sending sample to the ui
	private Random rand = new Random(System.currentTimeMillis());
	private Dianne dianne;
	private DiannePlatform platform;
	private DianneDatasets datasets;
	
	// can be used for timestamping, but won't always work (i.e. when multiple sources trigger inputs at the same time)
	// works for basic demo purposes though
	private long start,stop;
	
	@Activate
	public void activate(BundleContext c){
		this.context = c;
		this.imageConverter = new ImageConverter();
	}
	
	@Reference
	void setDianne(Dianne d){
		dianne = d;
	}
	
	@Reference
	void setDiannePlatform(DiannePlatform p){
		platform = p;
	}
	
	@Reference(cardinality=ReferenceCardinality.OPTIONAL)
	void setDianneDatasets(DianneDatasets d){
		datasets = d;
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
		
		String moduleId = request.getParameter("moduleId");
		
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
				SSEForwardListener listener = new SSEForwardListener(nnId, moduleId, labels, request.startAsync());
				listener.register(context);
			} catch(Exception e){
				e.printStackTrace();
			}
		}
	}
	
	@Override
	protected void doPost(HttpServletRequest request, HttpServletResponse response)
			throws ServletException, IOException {
		response.setContentType("application/json");
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
		
		NeuralNetwork nn = null;
		try {
			nn = dianne.getNeuralNetwork(nni).getValue();
		} catch (Exception e) {
		}
		if(nn==null){
			System.out.println("Neural network instance "+id+" not available");
			return;
		}
		
		
		if(request.getParameter("forward")!=null){
			String inputId = request.getParameter("input");
			
			JsonObject sample = parser.parse(request.getParameter("forward")).getAsJsonObject();
			Tensor t = jsonConverter.fromJson(sample);
			
			start = System.currentTimeMillis();
			nn.forward(UUID.fromString(inputId), null, t, "ui");
			
		} else if(request.getParameter("url")!=null){
			String url = request.getParameter("url");
			String inputId = request.getParameter("input");
			
			Tensor t = null;
			try {
				URL u = new URL(url);
				BufferedImage img = ImageIO.read(u);
				t = imageConverter.fromImage(img);
			} catch(Exception e){
				System.out.println("Failed to read image from url "+url);
				return;
			}
			
			start = System.currentTimeMillis();
			nn.forward(UUID.fromString(inputId), null, t, "ui");
			
		} else if(request.getParameter("mode")!=null){
			String mode = request.getParameter("mode");
			String targetId = request.getParameter("target");

			Module m = nn.getModules().get(UUID.fromString(targetId));
			if(m!=null){
				m.setMode(EnumSet.of(Mode.valueOf(mode)));
			}
		} else if(request.getParameter("dataset")!=null){
			String dataset = request.getParameter("dataset");
			if(datasets == null){
				System.out.println("Datasets service not available");
				return;
			}
			Dataset d = datasets.getDataset(dataset);
			if(d == null){
				System.out.println("Dataset "+dataset+" not available");
				return;
			}

			Sample s = d.getSample(rand.nextInt(d.size()));

			String inputId = request.getParameter("input");

			if(inputId != null){
				start = System.currentTimeMillis();
				nn.forward(UUID.fromString(inputId), null, s.input, "ui");
			}
			
			JsonObject sample = jsonConverter.toJson(s.input);			
			String[] labels = d.getLabels();
			if(labels != null){
				sample.add("target", new JsonPrimitive(labels[TensorOps.argmax(s.target)]));
			} else {
				if(s.target.size() < 10){
					JsonObject target = jsonConverter.toJson(s.target);
					sample.add("target", target.get("data").getAsJsonArray());			
				}
			}
			
			response.getWriter().println(sample.toString());
			response.getWriter().flush();
		}
	}
	
	private String outputSSEMessage(UUID outputId, String[] outputLabels, Tensor output, long time, String...tags){
		JsonObject data = jsonConverter.toJson(output);

		float max = TensorOps.max(output);
		if(outputLabels != null || isProbability(output)){
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
			
			JsonArray probabilities = new JsonArray();
			for(int i=0;i<values.length;i++){
				// if negative values for classification - assume log probabilities
				// take exp to return probability
				if(max < 0){
					values[i] = (float)Math.exp(values[i]);
				}
				probabilities.add(new JsonPrimitive(values[i]));
			}
			data.add("probabilities", probabilities);
			
			JsonArray l = new JsonArray();
			for(int i=0;i<labels.length;i++){
				l.add(new JsonPrimitive(labels[i]));
			}
			data.add("labels", l);
		}

		if(time > 0){
			data.add("time", new JsonPrimitive(time));
		}

		if(tags!=null){
			JsonArray ta = new JsonArray();
			for(String tt : tags){
				if(tt.equals("ui") || tt.startsWith("_")) // ignore the ui tag
					continue;
				
				ta.add(new JsonPrimitive(tt));
			}
			data.add("tags",ta);
		}
		
		data.add("id", new JsonPrimitive(outputId.toString()));
		
		StringBuilder builder = new StringBuilder();
		builder.append("data: ").append(data.toString()).append("\n\n");
		return builder.toString();
	}
	
	private boolean isProbability(Tensor t){
		// estimate whether the tensor represents probabilties (== sums up to 1 or exps sum up to one)
		if(Math.abs(1.0f - TensorOps.sum(t)) < 0.0001){
			return true;
		}
		
		if(Math.abs(1.0f - TensorOps.sum(TensorOps.exp(null, t))) < 0.0001){
			return true;
		}
		
		return false;
	}
	
	private String errorSSEMessage(ModuleException exception){
		JsonObject data = new JsonObject();
		
		data.add("id", new JsonPrimitive(exception.moduleId.toString()));
		data.add("error", new JsonPrimitive(exception.getMessage()));
		
		StringBuilder builder = new StringBuilder();
		builder.append("data: ").append(data.toString()).append("\n\n");
		return builder.toString();
	}
	
	private class SSEForwardListener implements ForwardListener {

		private final String nnId;
		private final String moduleId;
		private final Map<UUID, String[]> labels;
		private final AsyncContext async;
		private ServiceRegistration<ForwardListener> reg;
		private Tensor copy = new Tensor();
		
		public SSEForwardListener(String nnId, String moduleId,
				Map<UUID, String[]> labels,
				AsyncContext async) {
			this.nnId = nnId;
			this.moduleId = moduleId;
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
			Dictionary<String, Object> props = new Hashtable<>();
			String target = nnId;
			if(moduleId != null)
				target+=":"+moduleId;
			props.put("targets", new String[]{target});
			props.put("aiolos.unique", true);
			reg = context.registerService(ForwardListener.class, this, props);
		}
		
		public void unregister(){
			if(reg != null){
				reg.unregister();
				reg = null;
			}
		}
		
		@Override
		public void onForward(UUID moduleId, Tensor output, String... tags) {
			stop = System.currentTimeMillis();
			try {
				// one can only measure time for samples that were forwarded from the UI
				long time = -1;
				if(tags!=null){
					for(String t : tags){
						if(t.equals("ui")){
							time = stop-start;
						}
					}
				}

				copy = output.copyInto(copy);
				String sseMessage = outputSSEMessage(moduleId, labels.get(moduleId), copy, time, tags);
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

		@Override
		public void onError(UUID moduleId, ModuleException ex, String... tags) {
			try {
				// only show exceptions for samples fired by UI 
				boolean show = false;
				if(tags!=null){
					for(String t : tags){
						if(t.equals("ui")){
							show = true;
						}
					}
				}
				if(show){
					String sseMessage = errorSSEMessage(ex);
					PrintWriter writer = async.getResponse().getWriter();
					writer.write(sseMessage);
					writer.flush();
				}

			} catch(Exception e){
			}	
		}
	}
}
