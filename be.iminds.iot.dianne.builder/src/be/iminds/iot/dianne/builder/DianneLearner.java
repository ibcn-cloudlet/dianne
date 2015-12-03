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
import be.iminds.iot.dianne.api.nn.eval.Evaluation;
import be.iminds.iot.dianne.api.nn.eval.Evaluator;
import be.iminds.iot.dianne.api.nn.learn.LearnProgress;
import be.iminds.iot.dianne.api.nn.learn.Learner;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkInstanceDTO;
import be.iminds.iot.dianne.api.nn.platform.DiannePlatform;
import be.iminds.iot.dianne.api.repository.DianneRepository;
import be.iminds.iot.dianne.api.repository.RepositoryListener;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.google.gson.JsonPrimitive;

@Component(service = { javax.servlet.Servlet.class }, 
	property = { "alias:String=/dianne/learner",
		 		 "osgi.http.whiteboard.servlet.pattern=/dianne/learner",
		 		 "osgi.http.whiteboard.servlet.asyncSupported:Boolean=true",
				 "aiolos.proxy=false" }, 
	immediate = true)
public class DianneLearner extends HttpServlet {

	private BundleContext context;
	private TensorFactory factory;
	private DianneRepository repository;
	
	private static final JsonParser parser = new JsonParser();
	
	private Map<String, Dataset> datasets = new HashMap<String, Dataset>();
	private DiannePlatform platform;
	private Learner learner;
	private Evaluator evaluator;

	private int interval = 10;
	private AsyncContext sse = null;
	
	@Activate
	void activate(BundleContext context){
		this.context = context;
	}
	
	@Reference
	void setTensorFactory(TensorFactory factory){
		this.factory = factory;
	}
	
	@Reference
	void setDianneRepository(DianneRepository repo){
		this.repository = repo;
	}
	
	@Reference
	void setLearner(Learner l){
		this.learner = l;
	}
	
	@Reference
	void setEvaluator(Evaluator e){
		this.evaluator = e;
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
			learner.stop();
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
					boolean clean = learnerConfig.get("clean").getAsBoolean();
					
					Map<String, String> config = new HashMap<>();
					config.put("startIndex", ""+start);
					config.put("endIndex", ""+end);
					config.put("batchSize", ""+batch);
					config.put("learningRate", ""+learningRate);
					config.put("momentum", ""+momentum);
					config.put("regularization", ""+regularization);
					config.put("criterion", criterion);
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
						
						learner.learn(nni, dataset, config);

					} catch (Exception e) {
						e.printStackTrace();
					}
				
				}else if(action.equals("evaluate")){
					int start = datasetConfig.get("train").getAsInt();
					int end = start+datasetConfig.get("test").getAsInt();
					
					Map<String, String> config = new HashMap<>();
					config.put("startIndex", ""+start);
					config.put("endIndex", ""+end);
					
					try {
						Evaluation result = evaluator.eval(nni, dataset, config);
						
						JsonObject eval = new JsonObject();
						eval.add("accuracy", new JsonPrimitive(result.accuracy()*100));
						
						Tensor confusionMatrix = result.getConfusionMatix();
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
						
						response.getWriter().write(eval.toString());
						response.getWriter().flush();
					} catch(Exception e){
						e.printStackTrace();
					}
				}
				break;
			}
		}
	}
	
	private class SSERepositoryListener implements RepositoryListener{

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
		
		@Override
		public void onParametersUpdate(UUID nnId, Collection<UUID> moduleIds,
				String... tag) {
			try {
				LearnProgress progress = learner.getProgress();
				if(progress.iteration == 0) // ignore if no progress yet
					return;
				
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
	
		public void register(BundleContext context){
			Dictionary<String, Object> props = new Hashtable();
			props.put("targets", new String[]{":"+nnId});
			props.put("aiolos.unique", true);
			reg = context.registerService(RepositoryListener.class, this, props);
		}
		
		public void unregister(){
			reg.unregister();
		}
		
	}
}
