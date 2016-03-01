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

import be.iminds.iot.dianne.api.coordinator.AgentResult;
import be.iminds.iot.dianne.api.coordinator.DianneCoordinator;
import be.iminds.iot.dianne.api.coordinator.EvaluationResult;
import be.iminds.iot.dianne.api.coordinator.LearnResult;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkDTO;
import be.iminds.iot.dianne.api.nn.platform.DiannePlatform;
import be.iminds.iot.dianne.nn.util.DianneCoordinatorWriter;
import be.iminds.iot.dianne.nn.util.DianneJSONConverter;

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
			e.printStackTrace();
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
		case "act":
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
			} else if(method.equals("eval")){
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
			} else if(method.equals("act")){
				Promise<AgentResult> result = null;
				if(nnName!=null){
					result= coordinator.act(nnName, dataset, config);
				} else {
					result = coordinator.act(nn, dataset, config);
				}
				result.then(p -> {
					writeResult(writer, id, null);
					return null;
				}, p -> {
					writeError(writer, id, -32603, "Error during learning: "+p.getFailure().getMessage());
				});
			}
			break;
		case "learnResult":
		case "evaluationResult":
		case "agentResult":
		case "job":
		case "stop":
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
			} else if(method.equals("agentResult")){
				writeResult(writer, id, coordinator.getAgentResult(jobId));
			} else if(method.equals("stop")) {
				try {
					coordinator.stop(jobId);
				} catch(Exception e){
					writeError(writer, id, -32603, "Error stopping job: "+e.getMessage());
				}
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
			DianneCoordinatorWriter.writeObject(writer, result);
		} catch(Throwable t){
			t.printStackTrace();
		}
		// end result object
		writer.endObject();
		writer.flush();			
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