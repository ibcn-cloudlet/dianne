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

import java.io.IOException;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.Collection;
import java.util.Dictionary;
import java.util.HashMap;
import java.util.Hashtable;
import java.util.Map;
import java.util.Map.Entry;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;

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

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.google.gson.JsonPrimitive;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.nn.eval.ClassificationEvaluation;
import be.iminds.iot.dianne.api.nn.eval.Evaluation;
import be.iminds.iot.dianne.api.nn.eval.Evaluator;
import be.iminds.iot.dianne.api.nn.learn.LearnProgress;
import be.iminds.iot.dianne.api.nn.learn.Learner;
import be.iminds.iot.dianne.api.nn.learn.LearnerListener;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkInstanceDTO;
import be.iminds.iot.dianne.api.nn.platform.DiannePlatform;
import be.iminds.iot.dianne.api.repository.DianneRepository;
import be.iminds.iot.dianne.api.repository.RepositoryListener;
import be.iminds.iot.dianne.tensor.Tensor;

@Component(service = { javax.servlet.Servlet.class }, 
	property = { "alias:String=/dianne/learner",
		 		 "osgi.http.whiteboard.servlet.pattern=/dianne/learner",
		 		 "osgi.http.whiteboard.servlet.asyncSupported:Boolean=true",
				 "aiolos.proxy=false" }, 
	immediate = true)
public class DianneLearner extends HttpServlet {

	private BundleContext context;
	private DianneRepository repository;
	
	private static final JsonParser parser = new JsonParser();
	
	private Map<String, Dataset> datasets = new HashMap<String, Dataset>();
	private DiannePlatform platform;
	private Map<String, Learner> learners = new ConcurrentHashMap<>();
	private Map<String, Evaluator> evaluators = new ConcurrentHashMap<>();

	private int interval = 10;
	private AsyncContext sse = null;
	
	@Activate
	void activate(BundleContext context){
		this.context = context;
	}
	
	@Reference
	void setDianneRepository(DianneRepository repo){
		this.repository = repo;
	}
	
	@Reference(cardinality=ReferenceCardinality.MULTIPLE, 
			policy=ReferencePolicy.DYNAMIC)
	void addLearner(Map<String, Object> properties, Learner l){
		String category = (String)properties.get("dianne.learner.category");
		this.learners.put(category, l);
	}
	
	void removeLearner(Learner l){
		this.learners.values().remove(l);
	}
	
	@Reference(cardinality=ReferenceCardinality.MULTIPLE, 
			policy=ReferencePolicy.DYNAMIC)
	void addEvaluator(Map<String, Object> properties, Evaluator e){
		String category = (String)properties.get("dianne.evaluator.category");
		this.evaluators.put(category, e);
	}
	
	void removeEvaluator(Evaluator e){
		this.evaluators.values().remove(e);
	}
	
	@Reference
	void setDiannePlatform(DiannePlatform p){
		platform = p;
	}
	
	@Reference(cardinality=ReferenceCardinality.AT_LEAST_ONE, 
			policy=ReferencePolicy.DYNAMIC)
	void addDataset(Dataset dataset, Map<String, Object> properties){
		String name = (String) properties.get("name");
		this.datasets.put(name, dataset);
	}
	
	void removeDataset(Dataset dataset, Map<String, Object> properties){
		String name = (String) properties.get("name");
		this.datasets.remove(name);
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
				SSERepositoryListener listener = new SSERepositoryListener(nnId, request.startAsync());
				listener.register(context);
			} catch(Exception e){
				e.printStackTrace();
			}
		}
		
//		sse = request.startAsync();
//		sse.setTimeout(0); // no timeout => remove listener when error occurs.
	}
	
	@Override
	protected void doPost(HttpServletRequest request, HttpServletResponse response)
			throws ServletException, IOException {
		String id = request.getParameter("id");
		if(id == null){
			System.out.println("No neural network instance specified");
		}
		UUID nnId = UUID.fromString(id);
		NeuralNetworkInstanceDTO nni = platform.getNeuralNetworkInstance(nnId);
		if(nni==null){
			System.out.println("Neural network instance "+id+" not deployed");
			return;
		}

		String action = request.getParameter("action");
		if(action.equals("stop")){
			learners.get("FF").stop();
			return;
		}
		
		String target = request.getParameter("target");
		// this list consists of all ids of modules that are needed for trainer:
		// the input, output, trainable and preprocessor modules
		String configJsonString = request.getParameter("config");
		
		JsonObject configJson = parser.parse(configJsonString).getAsJsonObject();
		
		JsonObject learnerConfig = (JsonObject) configJson.get(target);
		
		for(Entry<String, JsonElement> configs : configJson.entrySet()){
			JsonObject datasetConfig = (JsonObject) configs.getValue();
			if(datasetConfig.get("category").getAsString().equals("Dataset")
					&& datasetConfig.get("input") !=null){
				String dataset = datasetConfig.get("dataset").getAsString();
				if(action.equals("learn")){
					int start = 0;
					int end = datasetConfig.get("train").getAsInt();
					
					int batch = learnerConfig.get("batch").getAsInt();
					float learningRate = learnerConfig.get("learningRate").getAsFloat();
					float momentum = learnerConfig.get("momentum").getAsFloat();
					float regularization = learnerConfig.get("regularization").getAsFloat();
					String criterion = learnerConfig.get("loss").getAsString();
					String method = learnerConfig.get("method").getAsString();

					boolean clean = learnerConfig.get("clean").getAsBoolean();
					
					Map<String, String> config = new HashMap<>();
					config.put("startIndex", ""+start);
					config.put("endIndex", ""+end);
					config.put("batchSize", ""+batch);
					config.put("learningRate", ""+learningRate);
					config.put("momentum", ""+momentum);
					config.put("regularization", ""+regularization);
					config.put("criterion", criterion);
					config.put("method", method);
					config.put("syncInterval", ""+interval);
					config.put("clean", clean ? "true" : "false");
					
					try {
						Dataset d = datasets.get(dataset);
						if(d!=null){
							JsonObject labels = new JsonObject();
							labels.add(target, new JsonPrimitive(Arrays.toString(d.getLabels())));
							response.getWriter().write(labels.toString());
							response.getWriter().flush();
						}
						
						learners.get("FF").learn(dataset, config, nni);

					} catch (Exception e) {
						e.printStackTrace();
					}
				
				}else if(action.equals("evaluate")){
					int start = datasetConfig.get("train").getAsInt();
					int end = start+datasetConfig.get("test").getAsInt();
					
					Map<String, String> config = new HashMap<>();
					config.put("startIndex", ""+start);
					config.put("endIndex", ""+end);
					
					String type = "REGRESSION";
					if(platform.isClassificationDatset(dataset)){
						type = "CLASSIFICATION";
					}
					
					try {
						Evaluation result = evaluators.get(type).eval(dataset, config, nni);

						JsonObject eval = new JsonObject();
						
						if(result instanceof ClassificationEvaluation){
							ClassificationEvaluation ce = (ClassificationEvaluation)result;
							eval.add("accuracy", new JsonPrimitive(ce.accuracy()*100));
							
							Tensor confusionMatrix = ce.getConfusionMatix();
							JsonArray data = new JsonArray();
							for(int i=0;i<confusionMatrix.size(0);i++){
								for(int j=0;j<confusionMatrix.size(1);j++){
									JsonArray element = new JsonArray();
									element.add(new JsonPrimitive(i));
									element.add(new JsonPrimitive(j));
									element.add(new JsonPrimitive(confusionMatrix.get(i,j)));
									data.add(element);
								}
							}
							eval.add("confusionMatrix", data);
						} else {
							eval.add("error", new JsonPrimitive(result.error()));
						}
						
						response.getWriter().write(eval.toString());
						response.getWriter().flush();
					} catch(Exception e){
						e.printStackTrace();
						JsonObject eval = new JsonObject();
						eval.add("error", new JsonPrimitive(e.getCause().getMessage()));
						response.getWriter().write(eval.toString());
						response.getWriter().flush();
					}
				}
				break;
			}
		}
	}
	
	private class SSERepositoryListener implements LearnerListener {

		private final String nnId;
		private final AsyncContext async;

		private ServiceRegistration reg;
		
		private int i = 0;
		
		public SSERepositoryListener(String nnId, AsyncContext async){
			this.nnId = nnId;
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
			props.put("targets", new String[]{learners.get("FF").getLearnerId().toString()});
			props.put("aiolos.unique", true);
			reg = context.registerService(LearnerListener.class, this, props);
		}
		
		public void unregister(){
			reg.unregister();
		}

		@Override
		public void onProgress(UUID learnerId, LearnProgress progress) {
			try {
				JsonObject data = new JsonObject();
				data.add("sample", new JsonPrimitive(progress.iteration));
				data.add("error", new JsonPrimitive(progress.error));
				StringBuilder builder = new StringBuilder();
				builder.append("data: ").append(data.toString()).append("\n\n");
				
				PrintWriter writer = async.getResponse().getWriter();
				writer.write(builder.toString());
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
		public void onException(UUID learnerId, Throwable ex) {
			try {
				PrintWriter writer = async.getResponse().getWriter();
	
				JsonObject err = new JsonObject();
				err.add("error", new JsonPrimitive(ex.getCause().getMessage()));
				StringBuilder builder = new StringBuilder();
				builder.append("data: ").append(err.toString()).append("\n\n");
				
				writer.write(builder.toString());
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
		public void onFinish(UUID learnerId, LearnProgress p) {
	
		}
		
	}
}
