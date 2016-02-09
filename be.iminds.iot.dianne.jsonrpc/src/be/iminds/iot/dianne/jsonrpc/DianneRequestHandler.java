package be.iminds.iot.dianne.jsonrpc;

import java.util.Map;
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
import be.iminds.iot.dianne.api.coordinator.LearnResult;
import be.iminds.iot.dianne.api.nn.eval.Evaluation;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkDTO;
import be.iminds.iot.dianne.nn.util.DianneJSONConverter;
import be.iminds.iot.dianne.tensor.Tensor;

@Component
public class DianneRequestHandler implements JSONRPCRequestHandler {

	private DianneCoordinator coordinator;
	
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
		if(method.equals("learn")
				|| method.equals("eval") ){
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
					writeLearnResult(writer, id, p.getValue());
					return null;
				}, p -> {
					writeError(writer, id, -32603, "Error during learning: "+p.getFailure().getMessage());
				});
			} else {
				// eval
				Promise<Evaluation> result = null;
				if(nnName!=null){
					result= coordinator.eval(nnName, dataset, config);
				} else {
					result = coordinator.eval(nn, dataset, config);
				}
				result.then(p -> {
					writeEvalResult(writer, id, p.getValue());
					return null;
				}, p -> {
					writeError(writer, id, -32603, "Error during learning: "+p.getFailure().getMessage());
				});
			}
			
		
		} else {
			writeError(writer, id, -32601, "Method "+method+" not found");
			return;
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
	
	private void writeLearnResult(JsonWriter writer, String id, LearnResult result) throws Exception{
		writer.beginObject();
		writer.name("jsonrpc");
		writer.value("2.0");
		writer.name("id");
		writer.value(id);
		writer.name("result");
		writer.beginArray();
		// write result object
		writer.beginObject();
		writer.name("error");
		writer.value(result.error);
		writer.name("iterations");
		writer.value(result.iterations);
		writer.endObject();
		// end result object
		writer.endArray();
		writer.endObject();
		writer.flush();			
	}
	
	private void writeEvalResult(JsonWriter writer, String id, Evaluation result) throws Exception{
		writer.beginObject();
		writer.name("jsonrpc");
		writer.value("2.0");
		writer.name("id");
		writer.value(id);
		writer.name("result");
		writer.beginArray();
		// write result object
		writer.beginObject();
		writer.name("accuracy");
		writer.value(result.accuracy());
		writer.name("forwardTime");
		writer.value(result.forwardTime());
		writer.name("outputs");
		writer.beginArray();
		for(Tensor t : result.getOutputs()){
			writer.beginArray();
			for(float f : t.get()){
				writer.value(f);
			}
			writer.endArray();
		}
		writer.endArray();
		writer.endObject();
		// end result object
		writer.endArray();
		writer.endObject();
		writer.flush();			
	}
	
	@Reference
	void setDianneCoordinator(DianneCoordinator c){
		this.coordinator = c;
	}
}
