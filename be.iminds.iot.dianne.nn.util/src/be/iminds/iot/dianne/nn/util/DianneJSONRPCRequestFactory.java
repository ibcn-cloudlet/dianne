package be.iminds.iot.dianne.nn.util;

import java.util.Map;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonPrimitive;

import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkDTO;

public class DianneJSONRPCRequestFactory {

	public static JsonObject createLearnRequest(int id, NeuralNetworkDTO nn, String dataset, Map<String, String> properties){
		JsonObject request = new JsonObject();
		request.add("jsonrpc", new JsonPrimitive("2.0"));
		request.add("method", new JsonPrimitive("learn"));
		request.add("id", new JsonPrimitive(id));
		
		JsonArray params = new JsonArray();
		params.add(DianneJSONConverter.toJson(nn));
		params.add(new JsonPrimitive(dataset));
		params.add(createJsonFromMap(properties));
		request.add("params", params);
		
		return request;
	}
	
	public static JsonObject createEvalRequest(int id, String nnName, String dataset, Map<String, String> properties){
		JsonObject request = new JsonObject();
		request.add("jsonrpc", new JsonPrimitive("2.0"));
		request.add("method", new JsonPrimitive("eval"));
		request.add("id", new JsonPrimitive(id));
		
		JsonArray params = new JsonArray();
		params.add(new JsonPrimitive(nnName));
		params.add(new JsonPrimitive(dataset));
		params.add(createJsonFromMap(properties));
		request.add("params", params);
		
		return request;
		
	}
	
	private static JsonObject createJsonFromMap(Map<String, String> map){
		JsonObject json = new JsonObject();
		map.entrySet().stream().forEach(e -> json.add(e.getKey(), new JsonPrimitive(e.getValue())));
		return json;
	}
	
}
