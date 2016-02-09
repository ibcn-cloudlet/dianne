package be.iminds.iot.dianne.jsonrpc;

import com.google.gson.JsonObject;
import com.google.gson.stream.JsonReader;
import com.google.gson.stream.JsonWriter;

public interface JSONRPCRequestHandler {

	void handleRequest(JsonReader reader, JsonWriter writer) throws Exception;

	void handleRequest(JsonObject request, JsonWriter writer) throws Exception;

}