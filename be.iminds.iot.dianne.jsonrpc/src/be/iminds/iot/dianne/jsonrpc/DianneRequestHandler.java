package be.iminds.iot.dianne.jsonrpc;

import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.stream.Collectors;

import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;
import org.osgi.util.promise.Promise;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.google.gson.stream.JsonReader;
import com.google.gson.stream.JsonWriter;

import be.iminds.iot.dianne.api.coordinator.DianneCoordinator;
import be.iminds.iot.dianne.api.coordinator.EvaluationResult;
import be.iminds.iot.dianne.api.coordinator.LearnResult;
import be.iminds.iot.dianne.api.nn.eval.Evaluation;
import be.iminds.iot.dianne.api.nn.eval.EvaluationProgress;
import be.iminds.iot.dianne.api.nn.learn.LearnProgress;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkDTO;
import be.iminds.iot.dianne.api.nn.platform.DiannePlatform;
import be.iminds.iot.dianne.nn.util.DianneJSONConverter;
import be.iminds.iot.dianne.tensor.Tensor;

@Component
public class DianneRequestHandler implements JSONRPCRequestHandler {

	private DianneCoordinator coordinator;
	private DiannePlatform platform;
	
	@Override
	public void handleRequest(JsonReader reader, JsonWriter writer) throws Exception {
		JsonParser parser = new JsonParser();
		JsonObject request = null; 
		try {
			request = parser.parse(reader).getAsJsonObject();
			handleRequest(request, writer);
		} catch(Exception e){
			writeError(writer, null, -32700, "Parse error");
			return;
		}
	}
	
	@Override
	public void handleRequest(JsonObject request, JsonWriter writer) throws Exception {
		String i = "null";
		if(request.has("id")){
			i = request.get("id").getAsString();
		}
		final String id = i;
		
		if(!request.has("jsonrpc")){
			writeError(writer, id, -32600, "Invalid JSONRPC request");
			return;
		}
		
		if(!request.get("jsonrpc").getAsString().equals("2.0")){
			writeError(writer, id, -32600, "Wrong JSONRPC version: "+request.get("jsonrpc").getAsString());
			return;
		}
		
		if(!request.has("method")){
			writeError(writer, id, -32600, "No method specified");
			return;
		}

		String method = request.get("method").getAsString();
		
		// TODO use a more generic approach here?
		switch(method){
		case "learn":
		case "eval":
			String nnName = null;
			NeuralNetworkDTO nn = null;
			String dataset;
			Map<String, String> config;
			
			try {
				JsonArray params = request.get("params").getAsJsonArray();
				if(params.get(0).isJsonPrimitive()){
					nnName = params.get(0).getAsString();
				} else {
					nn = DianneJSONConverter.parseJSON(params.get(0).getAsJsonObject());
				}
				dataset = params.get(1).getAsString();
				config = params.get(2).getAsJsonObject()
						.entrySet().stream().collect(Collectors.toMap( e -> e.getKey(), e -> e.getValue().getAsString()));

			} catch(Exception e){
				writeError(writer, id, -32602, "Incorrect parameters provided: "+e.getMessage());
				return;
			}
			
			// call coordinator
			if(method.equals("learn")){
				// learn
				Promise<LearnResult> result = null;
				if(nnName!=null){
					result= coordinator.learn(nnName, dataset, config);
				} else {
					result = coordinator.learn(nn, dataset, config);
				}
				result.then(p -> {
					writeResult(writer, id, p.getValue());
					return null;
				}, p -> {
					writeError(writer, id, -32603, "Error during learning: "+p.getFailure().getMessage());
				});
			} else {
				// eval
				Promise<EvaluationResult> result = null;
				if(nnName!=null){
					result= coordinator.eval(nnName, dataset, config);
				} else {
					result = coordinator.eval(nn, dataset, config);
				}
				result.then(p -> {
					writeResult(writer, id, p.getValue());
					return null;
				}, p -> {
					writeError(writer, id, -32603, "Error during learning: "+p.getFailure().getMessage());
				});
			}
			break;
		case "learnResult":
		case "evaluationResult":
		case "job":
			UUID jobId = null;
			try {
				JsonArray params = request.get("params").getAsJsonArray();
				if(params.get(0).isJsonPrimitive()){
					String s = params.get(0).getAsString();
					jobId = UUID.fromString(s);
				} 
			} catch(Exception e){
				writeError(writer, id, -32602, "Incorrect parameters provided: "+e.getMessage());
				return;
			}
			
			if(method.equals("learnResult")){
				writeResult(writer, id, coordinator.getLearnResult(jobId));
			} else if(method.equals("evaluationResult")){
				writeResult(writer, id, coordinator.getEvaluationResult(jobId));
			} else {
				writeResult(writer, id, coordinator.getJob(jobId));
			}
			break;	
		case "availableNeuralNetworks": 
			writeResult(writer, id, platform.getAvailableNeuralNetworks());
			break;
		case "availableDatasets":
			writeResult(writer, id, platform.getAvailableDatasets());
			break;
		case "queuedJobs":
			writeResult(writer, id, coordinator.queuedJobs());
			break;
		case "runningJobs":
			writeResult(writer, id, coordinator.runningJobs());
			break;
		case "finishedJobs":
			writeResult(writer, id, coordinator.finishedJobs());
			break;
		case "notifications":
			writeResult(writer, id, coordinator.getNotifications());
			break;	
		case "status":
			writeResult(writer, id, coordinator.getStatus());
			break;
		case "devices":
			writeResult(writer, id, coordinator.getDevices());
			break;
		default:
			writeError(writer, id, -32601, "Method "+method+" not found");
		}
	}
	
	private void writeError(JsonWriter writer, String id, int code, String message) throws Exception {
		writer.beginObject();
		writer.name("jsonrpc");
		writer.value("2.0");
		writer.name("id");
		writer.value(id);
		writer.name("error");
		writer.beginObject();
		// error object
		writer.name("code");
		writer.value(code);
		writer.name("message");
		writer.value(message);
		writer.endObject();
		// end error object
		writer.endObject();
		writer.flush();					
	}
	
	private void writeResult(JsonWriter writer, String id, Object result) throws Exception{
		writer.beginObject();
		writer.name("jsonrpc");
		writer.value("2.0");
		writer.name("id");
		writer.value(id);
		writer.name("result");
		// write result object
		try {
		writeObject(writer, result);
		} catch(Throwable t){
			t.printStackTrace();
		}
		// end result object
		writer.endObject();
		writer.flush();			
	}
	
	public void writeObject(JsonWriter writer, Object o) throws Exception {
		if(o==null){
			writer.value("null");
		} else if(o instanceof LearnResult){
			writeLearnResult(writer, (LearnResult) o);
		} else if(o instanceof EvaluationResult){
			writeEvaluationResult(writer, (EvaluationResult) o);
		} else if(o instanceof List){
			List l = (List)o;
			writer.beginArray();
			for(Object ll : l){
				writeObject(writer, ll);
			}
			writer.endArray();
		} else if(o instanceof Map){
			Map m = (Map) o;
			writer.beginObject();
			for(Object k : m.keySet()){
				writer.name(k.toString());
				writeObject(writer, m.get(k));
			}
			writer.endObject();
		} else if(o.getClass().equals(String.class)
				|| o.getClass().isEnum()
				|| o.getClass().equals(UUID.class)){
			writer.value(o.toString());
		} else {
			writer.beginObject();
			for(Field f : o.getClass().getFields()){
				if(Modifier.isPublic(f.getModifiers())){
					writer.name(f.getName());
					if(f.getType().isPrimitive()){ 
						switch(f.getType().getName()){
						case "long":
							writer.value(f.getLong(o));
							break;
						case "int":
							writer.value(f.getInt(o));
							break;
						case "float":
							writer.value(f.getFloat(o));
							break;
						case "double":
							writer.value(f.getDouble(o));
							break;
						case "boolean":
							writer.value(f.getBoolean(o));
							break;
						case "short": 
							writer.value(f.getShort(o));
							break;
						case "byte":
							writer.value(f.getByte(o));
							break;
						}
					} else {
						writeObject(writer, f.get(o));
					}
				}
			}
			writer.endObject();
		}
	}
	
	// dedicated methods for writing LearnResult and EvaluationResult objects ... these are no simple dtos
	private void writeLearnResult(JsonWriter writer, LearnResult result) throws Exception{
		writer.beginArray();
		for(LearnProgress p : result.progress){
			writer.beginObject();
			writer.name("error");
			writer.value(new Float(p.error));
			writer.name("iteration");
			writer.value(p.iteration);
			writer.endObject();		
		}
		writer.endArray();
	}
	
	private void writeEvaluationResult(JsonWriter writer, EvaluationResult result) throws Exception{
		writer.beginArray();
		for(Evaluation eval : result.evaluations.values()){
			writer.beginObject();
			if(eval==null){
				// write nothing?
			} else if(eval instanceof EvaluationProgress){
				writer.name("processed");
				writer.value(((EvaluationProgress) eval).getProcessed());
				writer.name("total");
				writer.value(((EvaluationProgress) eval).getTotal());
			} else {
				writer.name("accuracy");
				writer.value(new Float(eval.accuracy()));
				writer.name("forwardTime");
				writer.value(eval.forwardTime());
				// write confusion matrix
				writer.name("confusionMatrix");
				writer.beginArray();
				Tensor confusionMatrix = eval.getConfusionMatix();
				for(int i=0;i<confusionMatrix.size(0);i++){
					for(int j=0;j<confusionMatrix.size(1);j++){
						writer.beginArray();
						writer.value(i);
						writer.value(j);
						writer.value(new Float(confusionMatrix.get(i, j)));
						writer.endArray();
					}
				}
				writer.endArray();
				// write all outputs
				writer.name("outputs");
				writer.beginArray();
				for(Tensor t : eval.getOutputs()){
					writer.beginArray();
					for(float f : t.get()){
						writer.value(new Float(f));
					}
					writer.endArray();
				}
				writer.endArray();
			} 
			writer.endObject();	
		}
		writer.endArray();
	}
	
	@Reference
	void setDianneCoordinator(DianneCoordinator c){
		this.coordinator = c;
	}

	@Reference
	void setDiannePlatform(DiannePlatform p){
		this.platform = p;
	}
}