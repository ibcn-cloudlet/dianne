package be.iminds.iot.dianne.builder;

import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import javax.servlet.AsyncContext;
import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;
import org.osgi.service.component.annotations.ReferenceCardinality;
import org.osgi.service.component.annotations.ReferencePolicy;

import be.iminds.iot.dianne.dataset.Dataset;
import be.iminds.iot.dianne.dataset.DatasetAdapter;
import be.iminds.iot.dianne.nn.module.Input;
import be.iminds.iot.dianne.nn.module.Module;
import be.iminds.iot.dianne.nn.module.Output;
import be.iminds.iot.dianne.nn.module.Preprocessor;
import be.iminds.iot.dianne.nn.module.Trainable;
import be.iminds.iot.dianne.nn.train.Criterion;
import be.iminds.iot.dianne.nn.train.Evaluation;
import be.iminds.iot.dianne.nn.train.Evaluator;
import be.iminds.iot.dianne.nn.train.Trainer;
import be.iminds.iot.dianne.nn.train.criterion.MSECriterion;
import be.iminds.iot.dianne.nn.train.criterion.NLLCriterion;
import be.iminds.iot.dianne.nn.train.eval.ArgMaxEvaluator;
import be.iminds.iot.dianne.nn.train.eval.EvalProgressListener;
import be.iminds.iot.dianne.nn.train.strategy.StochasticGradient;
import be.iminds.iot.dianne.nn.train.strategy.TrainProgressListener;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.google.gson.JsonPrimitive;

@Component(service = { javax.servlet.Servlet.class }, 
	property = { "alias:String=/dianne/learner","aiolos.proxy=false" }, 
	immediate = true)
public class DianneLearner extends HttpServlet {

	private TensorFactory factory;
	
	private static final JsonParser parser = new JsonParser();
	
	private Map<String, Module> modules = new HashMap<String, Module>();
	private Map<String, Dataset> datasets = new HashMap<String, Dataset>();

	private AsyncContext sse = null;
	
	@Reference
	public void setTensorFactory(TensorFactory factory){
		this.factory = factory;
	}
	
	@Reference(cardinality=ReferenceCardinality.MULTIPLE, 
			policy=ReferencePolicy.DYNAMIC)
	public void addModule(Module m){
		this.modules.put(m.getId().toString(), m);
	}
	
	public void removeModule(Module m){
		Iterator<Entry<String, Module>> it = modules.entrySet().iterator();
		while(it.hasNext()){
			Entry<String, Module> e = it.next();
			if(e.getValue()==m){
				it.remove();
			}
		}
	}
	
	@Reference(cardinality=ReferenceCardinality.AT_LEAST_ONE, 
			policy=ReferencePolicy.DYNAMIC)
	public void addDataset(Dataset dataset){
		this.datasets.put(dataset.getName(), dataset);
	}
	
	public void removeDataset(Dataset dataset){
		Iterator<Entry<String, Dataset>> it = datasets.entrySet().iterator();
		while(it.hasNext()){
			Entry<String, Dataset> e = it.next();
			if(e.getValue()==dataset){
				it.remove();
			}
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
		// TODO check if parameters exist and are correct!
		String action = request.getParameter("action");
		String target = request.getParameter("target");
		String modulesJsonString = request.getParameter("modules");
		String configJsonString = request.getParameter("config");
		
		JsonObject configJson = parser.parse(configJsonString).getAsJsonObject();
		
		JsonObject processorConfig = (JsonObject) configJson.get(target);
		
		for(Entry<String, JsonElement> configs : configJson.entrySet()){
			JsonObject datasetConfig = (JsonObject) configs.getValue();
			if(datasetConfig.get("category").getAsString().equals("Dataset")
					&& datasetConfig.get("input") !=null){
				// found a dataset
				// TODO what in case of multiple datasets?
				if(action.equals("learn")){
					Dataset trainSet = createTrainDataset(datasetConfig);
					Trainer trainer = createTrainer(processorConfig);
					Criterion loss = createLoss(processorConfig);
					
					// TODO check if modules are present
					JsonArray moduleIds = parser.parse(modulesJsonString).getAsJsonArray();
					
					Input input = null;
					Output output = null;
					List<Trainable> trainable = new ArrayList<Trainable>();
					List<Preprocessor> preprocessors = new ArrayList<Preprocessor>();
					
					List<Module> toTrain = new ArrayList<Module>();
					Iterator<JsonElement> it = moduleIds.iterator();
					while(it.hasNext()){
						String id = it.next().getAsString();
						Module m = modules.get(id);
						if(m instanceof Input){
							// only include the input module connected to the dataset
							if(datasetConfig.get("input").getAsString().equals(id)){
								input = (Input) m;
								toTrain.add(m);
							}
						} else if(m instanceof Output){
							// only include the output module connected to the trainer
							if(processorConfig.get("output").getAsString().equals(id)){
								output = (Output) m;
								toTrain.add(m);
							}
						} else {
							if(m instanceof Trainable){
								trainable.add((Trainable) m);
							} else if(m instanceof Preprocessor){
								preprocessors.add((Preprocessor) m);
							}
							
							toTrain.add(m);
						}
					}
					trainer.train(toTrain, loss, trainSet);
					
					JsonObject parameters = new JsonObject();
					for(Trainable t : trainable){
						parameters.add(((Module)t).getId().toString(), new JsonPrimitive(Arrays.toString(t.getParameters().get())));;
					}
					for(Preprocessor p : preprocessors){
						parameters.add(((Module)p).getId().toString(), new JsonPrimitive(Arrays.toString(p.getParameters().get())));;
					}
					parameters.add(output.getId().toString(), new JsonPrimitive(Arrays.toString(output.getOutputLabels())));
					
					response.getWriter().write(parameters.toString());
					response.getWriter().flush();
				}else if(action.equals("evaluate")){
					Dataset testSet = createTestDataset(datasetConfig);
					Evaluator e = createEvaluator(processorConfig);
					
					Input input = (Input) modules.get(datasetConfig.get("input").getAsString());
					Output output = (Output) modules.get(processorConfig.get("output").getAsString());
					
					// Evaluate
					Evaluation result = e.evaluate(input, output, testSet);
					
					JsonObject eval = new JsonObject();
					eval.add("accuracy", new JsonPrimitive(result.accuracy()*100));
					response.getWriter().write(eval.toString());
					response.getWriter().flush();
				}
				break;
			}
		}
	}
	
	private Evaluator createEvaluator(JsonObject evaluatorConfig){
		ArgMaxEvaluator evaluator = new ArgMaxEvaluator(factory);
		evaluator.addProgressListener(new EvalProgressListener() {
			
			@Override
			public void onProgress(Tensor confusionMatrix) {
				if(sse!=null){
					try {
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
						
						StringBuilder builder = new StringBuilder();
						builder.append("data: ").append(data.toString()).append("\n\n");
		
						PrintWriter writer = sse.getResponse().getWriter();
						writer.write(builder.toString());
						writer.flush();
					} catch(Exception e){
						e.printStackTrace();
					}
				}
			}
		});
		return evaluator;
	}
	
	private Criterion createLoss(JsonObject trainerConfig){
		Criterion criterion = null;
		String loss = trainerConfig.get("loss").getAsString();
		if(loss.equals("MSE")){
			criterion = new MSECriterion(factory);
		} else if(loss.equals("NLL")){
			criterion = new NLLCriterion(factory);
		}
		return criterion;
	}
	
	private Trainer createTrainer(JsonObject trainerConfig){
		int batch = trainerConfig.get("batch").getAsInt();
		int epochs = trainerConfig.get("epochs").getAsInt();
		float learningRate = trainerConfig.get("learningRate").getAsFloat();
		float learningRateDecay = trainerConfig.get("learningRateDecay").getAsFloat();
		StochasticGradient trainer = new StochasticGradient(batch, epochs, learningRate, learningRateDecay);
		
		trainer.addProgressListener(new TrainProgressListener() {
			
			@Override
			public void onProgress(int epoch, int sample, float error) {
				if(sse!=null){
					try {
						JsonObject data = new JsonObject();
						data.add("epoch", new JsonPrimitive(epoch));
						data.add("sample", new JsonPrimitive(sample));
						data.add("error", new JsonPrimitive(error));
						StringBuilder builder = new StringBuilder();
						builder.append("data: ").append(data.toString()).append("\n\n");

						PrintWriter writer = sse.getResponse().getWriter();
						writer.write(builder.toString());
						writer.flush();
					} catch(Exception e){
						e.printStackTrace();
					}
				}
			}
		});
		return trainer;
	}
	
	private Dataset createTestDataset(JsonObject datasetConfig){
		// TODO check if dataset exists?
		String dataset = datasetConfig.get("dataset").getAsString();
		Dataset d = datasets.get(dataset);
		int start = datasetConfig.get("train").getAsInt();
		int end = start+datasetConfig.get("test").getAsInt();
		return new DatasetAdapter(d, start, end);
	}
	
	private Dataset createTrainDataset(JsonObject datasetConfig){
		// TODO check if dataset exists?
		String dataset = datasetConfig.get("dataset").getAsString();
		Dataset d = datasets.get(dataset);
		int start = 0;
		int end = datasetConfig.get("train").getAsInt();
		return new DatasetAdapter(d, start, end);
	}
}
