package be.iminds.iot.dianne.nn.runtime.util;

import java.util.ArrayList;
import java.util.Dictionary;
import java.util.Hashtable;
import java.util.Iterator;
import java.util.List;
import java.util.Map.Entry;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

public class DianneJSONParser {

	private static final JsonParser parser = new JsonParser();
	
	/**
	 * Parse a json string and return a list of properties that can be used
	 * by the ModuleManager to deploy modules
	 */
	public static List<Dictionary<String, Object>> parseJSON(String json){
		List<Dictionary<String, Object>> moduleConfigs = new ArrayList<Dictionary<String,Object>>();
		
		JsonObject modulesJson = parser.parse(json).getAsJsonObject();
		
		for(Entry<String, JsonElement> module : modulesJson.entrySet()){
			JsonObject moduleJson = (JsonObject) module.getValue();
			moduleConfigs.add(parseModuleJSON(moduleJson));
		}
		
		return moduleConfigs;
	}
	
	public static Dictionary<String, Object> parseModuleJSON(String module){
		JsonObject moduleJson = parser.parse(module).getAsJsonObject();
		return parseModuleJSON(moduleJson);
	}
	
	private static Dictionary<String, Object> parseModuleJSON(JsonObject moduleJson){
		Dictionary<String, Object> config = new Hashtable<String, Object>();
		
		
		String id = moduleJson.get("id").getAsString();
		String type = moduleJson.get("type").getAsString();
		
		// TODO standardize a configuration key-value format and key names
		config.put("module.id", id);
		config.put("module.type", type);
		
		if(moduleJson.has("next")){
			String next;
			if(moduleJson.get("next").isJsonArray()){
				JsonArray nextJson = moduleJson.get("next").getAsJsonArray();
				next = "";
				Iterator<JsonElement> it = nextJson.iterator();
				while(it.hasNext()){
					JsonElement e = it.next();
					next+=e.getAsString();
					if(it.hasNext()){
						next+=",";
					}
				}
			} else {
				next = moduleJson.get("next").getAsString();
			}
			config.put("module.next", next);
		}
		if(moduleJson.has("prev")){
			String prev;
			if(moduleJson.get("prev").isJsonArray()){
				JsonArray prevJson = moduleJson.get("prev").getAsJsonArray();
				prev = "";
				Iterator<JsonElement> it = prevJson.iterator();
				while(it.hasNext()){
					JsonElement e = it.next();
					prev+=e.getAsString();
					if(it.hasNext()){
						prev+=",";
					}
				}
			} else {
				prev = moduleJson.get("prev").getAsString();
			}
			config.put("module.prev", prev);
		}
		if(moduleJson.has("parameters")){
			config.put("module.parameters", moduleJson.get("parameters").getAsString());
		}

		// key prefix
		String prefix = "module."+type.toLowerCase()+"."; 
			
		for(Entry<String, JsonElement> property : moduleJson.entrySet()){
			String key = property.getKey();
			if(key.equals("id")
				|| key.equals("type")
				|| key.equals("prev")
				|| key.equals("next")){
				continue;
				// this is only for module-specific properties
			}
			// TODO already infer type here?
			config.put(prefix+property.getKey(), property.getValue().getAsString());
		}
		
		return config;
	}
}
