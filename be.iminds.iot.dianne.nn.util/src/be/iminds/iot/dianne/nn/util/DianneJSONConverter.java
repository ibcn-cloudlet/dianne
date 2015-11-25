package be.iminds.iot.dianne.nn.util;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.UUID;

import be.iminds.iot.dianne.api.nn.module.dto.ModuleDTO;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkDTO;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.google.gson.JsonPrimitive;

/**
 * JSON Parser class ... reads in either new or old JSON format and convert it to the DTOs
 * 
 * Should be replaced by the default DTO - JSON conversion in the end
 * 
 * @author tverbele
 *
 */
public class DianneJSONConverter {

	private static final JsonParser parser = new JsonParser();
	
	public static NeuralNetworkDTO parseJSON(String s){
		JsonObject json = parser.parse(s).getAsJsonObject();
		return parseJSON(json);
	}
	
	public static NeuralNetworkDTO parseJSON(JsonObject json){
		String name = null;
		List<ModuleDTO> modules = new ArrayList<ModuleDTO>();

		if(json.has("name")){
			name = json.get("name").getAsString();
		}
		
		// could be either a nice NeuralNetworkDTO or just a bunch of modules
		JsonObject jsonModules = json;
		if(json.has("modules")){
			jsonModules = json.get("modules").getAsJsonObject();
		}
		
		for(Entry<String, JsonElement> module : jsonModules.entrySet()){
			JsonObject moduleJson = (JsonObject) module.getValue();
			modules.add(parseModuleJSON(moduleJson));
		}
		
		return new NeuralNetworkDTO(name, modules);
	}
	
	
	
	public static ModuleDTO parseModuleJSON(String module){
		JsonObject moduleJson = parser.parse(module).getAsJsonObject();
		return parseModuleJSON(moduleJson);
	}
	
	private static ModuleDTO parseModuleJSON(JsonObject jsonModule){
		
		UUID id = UUID.fromString(jsonModule.get("id").getAsString());
		String type = jsonModule.get("type").getAsString();
		
		UUID[] next = null,prev = null;
		Map<String, String> properties = new HashMap<String, String>();
		
		if(jsonModule.has("next")){
			if(jsonModule.get("next").isJsonArray()){
				JsonArray jsonNext = jsonModule.get("next").getAsJsonArray();
			
				next = new UUID[jsonNext.size()];
				int i=0;
				Iterator<JsonElement> it = jsonNext.iterator();
				while(it.hasNext()){
					JsonElement e = it.next();
					next[i++] = UUID.fromString(e.getAsString());
				}
			} else {
				next = new UUID[1];
				next[0] = UUID.fromString(jsonModule.get("next").getAsString());
			}
		}
		if(jsonModule.has("prev")){
			if(jsonModule.get("prev").isJsonArray()){
				JsonArray jsonPrev = jsonModule.get("prev").getAsJsonArray();

				prev = new UUID[jsonPrev.size()];
				int i=0;
				Iterator<JsonElement> it = jsonPrev.iterator();
				while(it.hasNext()){
					JsonElement e = it.next();
					prev[i++] = UUID.fromString(e.getAsString());
				}
			} else {
				prev = new UUID[1];
				prev[0] = UUID.fromString(jsonModule.get("prev").getAsString());
			}
		}

// TODO this uses the old model where properties where just stored as flatmap		
		for(Entry<String, JsonElement> property : jsonModule.entrySet()){
			String key = property.getKey();
			if(key.equals("id")
				|| key.equals("type")
				|| key.equals("prev")
				|| key.equals("next")){
				continue;
				// this is only for module-specific properties
			}
			properties.put(property.getKey(), property.getValue().getAsString());
		}

// TODO evolve to a separate "properties" item
		if(jsonModule.has("properties")){
			JsonObject jsonProperties = jsonModule.get("properties").getAsJsonObject();
			for(Entry<String, JsonElement> jsonProperty : jsonProperties.entrySet()){
				String key = jsonProperty.getKey();
				String value = jsonProperty.getValue().getAsString();
				
				properties.put(key, value);
			}
		}
		
		ModuleDTO dto = new ModuleDTO(id, type, next, prev, properties);
		return dto;
	}
	
	public static String toJsonString(NeuralNetworkDTO dto){
		return toJsonString(dto, false);
	}

	public static String toJsonString(NeuralNetworkDTO dto, boolean pretty){
		JsonObject nn = toJson(dto);
	
		GsonBuilder builder= new GsonBuilder();
		if(pretty){
			builder.setPrettyPrinting();
		}
		Gson gson = builder.create();
		String output = gson.toJson(nn);
		
		return output;
	}
	
	private static JsonObject toJson(NeuralNetworkDTO dto){
		JsonObject nn = new JsonObject();
		
		JsonObject modules = new JsonObject();
		for(ModuleDTO m : dto.modules){
			JsonObject module = toJson(m);
			modules.add(m.id.toString(), module);
		}
		
		String name = dto.name==null ? "unnamed" : dto.name;
		nn.add("name", new JsonPrimitive(name));
		nn.add("modules", modules);
		return nn;
	}
	
	private static JsonObject toJson(ModuleDTO dto){
		JsonObject module = new JsonObject();
		
		module.add("id", new JsonPrimitive(dto.id.toString()));
		module.add("type", new JsonPrimitive(dto.type));

		if(dto.next!=null){
			JsonArray next = new JsonArray();
			for(UUID n : dto.next){
				next.add(new JsonPrimitive(n.toString()));
			}
			module.add("next", next);
		}
		
		if(dto.prev!=null){
			JsonArray prev = new JsonArray();
			for(UUID p : dto.prev){
				prev.add(new JsonPrimitive(p.toString()));
			}
			module.add("prev", prev);
		}
		
		if(dto.properties!=null){
			for(String k : dto.properties.keySet()){
				module.add(k, new JsonPrimitive(dto.properties.get(k)));
			}
		}
		
		return module;
	}
	
}
