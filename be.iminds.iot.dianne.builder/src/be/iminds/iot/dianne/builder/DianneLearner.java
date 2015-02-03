package be.iminds.iot.dianne.builder;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map.Entry;

import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;
import org.osgi.service.http.HttpService;

import be.iminds.iot.dianne.dataset.Dataset;
import be.iminds.iot.dianne.dataset.DatasetAdapter;
import be.iminds.iot.dianne.dataset.mnist.MNISTDataset;
import be.iminds.iot.dianne.nn.module.Input;
import be.iminds.iot.dianne.nn.module.Output;
import be.iminds.iot.dianne.nn.module.Trainable;
import be.iminds.iot.dianne.nn.train.Criterion;
import be.iminds.iot.dianne.nn.train.Evaluation;
import be.iminds.iot.dianne.nn.train.Evaluator;
import be.iminds.iot.dianne.nn.train.Trainer;
import be.iminds.iot.dianne.nn.train.criterion.MSECriterion;
import be.iminds.iot.dianne.nn.train.eval.ArgMaxEvaluator;
import be.iminds.iot.dianne.nn.train.strategy.StochasticGradient;
import be.iminds.iot.dianne.tensor.TensorFactory;

import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.google.gson.JsonPrimitive;

@Component(service = { javax.servlet.Servlet.class }, 
	property = { "alias:String=/dianne/learner","aiolos.proxy=false" }, 
	immediate = true)
public class DianneLearner extends HttpServlet {

	private TensorFactory factory;
	
	// for now fixed 1 input, 1 output, and trainable modules
	private Input input;
	private Output output;
	private List<Trainable> trainable = new ArrayList<Trainable>();
	
	@Reference
	public void setTensorFactory(TensorFactory factory){
		this.factory = factory;
	}
	
	@Reference
	public void setInput(Input input){
		this.input = input;
	}
	
	@Reference
	public void setOutput(Output output){
		this.output = output;
	}
	
	@Reference
	public void addTrainable(Trainable t){
		this.trainable.add(t);
	}
	
	@Reference
	public void setHttpService(HttpService http){
		try {
			// Use manual registration - problems with whiteboard
			http.registerServlet("/dianne/learner", this, null, null);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	@Override
	protected void doPost(HttpServletRequest request, HttpServletResponse response)
			throws ServletException, IOException {
		// TODO check if parameters exist and are correct!
		
		String action = request.getParameter("action");
		String target = request.getParameter("target");
		String configJsonString = request.getParameter("config");
		
		JsonObject configJson = new JsonParser().parse(configJsonString).getAsJsonObject();
	
		JsonObject processorConfig = (JsonObject) configJson.get(target);
		
		for(Entry<String, JsonElement> configs : configJson.entrySet()){
			JsonObject datasetConfig = (JsonObject) configs.getValue();
			if(datasetConfig.get("type").getAsString().equals("Dataset")
					&& datasetConfig.get("input") !=null){
				// found a dataset
				// TODO what in case of multiple datasets?
				if(action.equals("learn")){
					learn(datasetConfig, processorConfig);
				}else if(action.equals("evaluate")){
					Evaluation result = evaluate(datasetConfig, processorConfig);
					JsonObject eval = new JsonObject();
					eval.add("accuracy", new JsonPrimitive(result.accuracy()));
					response.getWriter().write(eval.toString());
					response.getWriter().flush();
				}
				break;
			}
		}
	}
	
	private void learn(JsonObject datasetConfig, JsonObject trainerConfig){
		Dataset trainSet = createTrainDataset(datasetConfig);
		Trainer trainer = createTrainer(trainerConfig);
		Criterion loss = createLoss(trainerConfig);
		
		// Train
		trainer.train(input, output, trainable, loss, trainSet);
	}
	
	private Evaluation evaluate(JsonObject datasetConfig, JsonObject evaluatorConfig){
		Dataset testSet = createTestDataset(datasetConfig);
		Evaluator eval = createEvaluator(evaluatorConfig);
		
		// Evaluate
		Evaluation result = eval.evaluate(input, output, testSet);
		return result;
	}
	
	private Evaluator createEvaluator(JsonObject evaluatorConfig){
		return new ArgMaxEvaluator(factory);
	}
	
	private Criterion createLoss(JsonObject trainerConfig){
		Criterion criterion = null;
		String loss = trainerConfig.get("loss").getAsString();
		if(loss.equals("MSE")){
			criterion = new MSECriterion(factory);
		} // TODO also support other criterions
		return criterion;
	}
	
	private Trainer createTrainer(JsonObject trainerConfig){
		int batch = trainerConfig.get("batch").getAsInt();
		int epochs = trainerConfig.get("epochs").getAsInt();
		Trainer trainer = new StochasticGradient(batch, epochs);
		return trainer;
	}
	
	private Dataset createTestDataset(JsonObject datasetConfig){
		Dataset dataset = createDataset(datasetConfig);
		int start = datasetConfig.get("train").getAsInt();
		int end = start+datasetConfig.get("test").getAsInt();
		return new DatasetAdapter(dataset, start, end);
	}
	
	private Dataset createTrainDataset(JsonObject datasetConfig){
		Dataset dataset = createDataset(datasetConfig);
		int start = 0;
		int end = datasetConfig.get("train").getAsInt();
		return new DatasetAdapter(dataset, start, end);
	}
	
	private Dataset mnist = null;
	
	private Dataset createDataset(JsonObject datasetConfig){
		String set = datasetConfig.get("dataset").getAsString();
		if(set.equals("MNIST")){
			if(mnist==null){
				 // TODO this should be done much better..
				mnist = new MNISTDataset(factory, "/home/tverbele/MNIST/");
			}
			return mnist;
		}
		return null;
	}
}
