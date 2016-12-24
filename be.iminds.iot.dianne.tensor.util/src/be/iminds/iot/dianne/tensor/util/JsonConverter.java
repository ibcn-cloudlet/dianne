package be.iminds.iot.dianne.tensor.util;

import java.util.Arrays;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.google.gson.JsonPrimitive;

import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;

public class JsonConverter {

	private JsonParser parser = new JsonParser();

	public JsonObject toJson(Tensor t){
		JsonObject json = new JsonObject();
		JsonArray dims = new JsonArray();
		for(int d : t.dims()){
			dims.add(new JsonPrimitive(d));
		}
		json.add("dims", dims);
		json.add("data", parser.parse(Arrays.toString(t.get())));
		
		// not really required, but handy to have full size available in js 
		json.add("size", new JsonPrimitive(t.size()));
		json.add("min", new JsonPrimitive(TensorOps.min(t)));
		json.add("max", new JsonPrimitive(TensorOps.max(t)));
		json.add("mean", new JsonPrimitive(TensorOps.mean(t)));
		return json;
	}
	
	public Tensor fromJson(JsonElement e){
		int[] dims = null;
		Tensor t = null;

		JsonArray dd = null;
		if(e.isJsonObject()){
			JsonObject json = e.getAsJsonObject();
			if(json.has("dims")){
				JsonArray d = json.get("dims").getAsJsonArray();
				dims = new int[d.size()];
				for(int i=0;i<dims.length;i++){
					dims[i] = d.get(i).getAsInt();
				}
			}	
			dd = json.get("data").getAsJsonArray();
		} else if(e.isJsonArray()){
			dd = e.getAsJsonArray();
		}
		
		if(dd==null || dd.size() == 0){
			// fill random data ?
			if(dims == null){
				t = new Tensor();
			} else {
				t = new Tensor(dims);
				t.randn();
			}
		} else {
			float[] data = parseData(dd.toString());
			if(dims == null){
				dims = new int[]{data.length};
			}
			if(data.length == 1){
				t = new Tensor(dims);
				t.fill(data[0]);
			} else {
				t = new Tensor(data, dims);
			}
		}
	
		return t;
	}
	
	public Tensor fromString(String jsonString){
		JsonElement json = parser.parse(jsonString);
		return fromJson(json);
	}
	
	private float[] parseData(String string){
		String[] strings = string.replace("[", "").replace("]", "").split(",");
		float result[] = new float[strings.length];
		for (int i = 0; i < result.length; i++) {
			result[i] = Float.parseFloat(strings[i]);
		}
		return result;
	}
}
