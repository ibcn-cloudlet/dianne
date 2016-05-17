/*******************************************************************************
 * DIANNE  - Framework for distributed artificial neural networks
 * Copyright (C) 2015  iMinds - IBCN - UGent
 *
 * This file is part of DIANNE.
 *
 * DIANNE is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * Contributors:
 *     Tim Verbelen, Steven Bohez
 *******************************************************************************/
package be.iminds.iot.dianne.nn.util;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

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
	
	public static JsonObject createDeployRequest(int id, String nnName){
		JsonObject request = new JsonObject();
		request.add("jsonrpc", new JsonPrimitive("2.0"));
		request.add("method", new JsonPrimitive("deploy"));
		request.add("id", new JsonPrimitive(id));
		
		JsonArray params = new JsonArray();
		params.add(new JsonPrimitive(nnName));
		request.add("params", params);
		
		return request;
	}
	
	public static JsonObject createUndeployRequest(int id, String nnId){
		JsonObject request = new JsonObject();
		request.add("jsonrpc", new JsonPrimitive("2.0"));
		request.add("method", new JsonPrimitive("undeploy"));
		request.add("id", new JsonPrimitive(id));
		
		JsonArray params = new JsonArray();
		params.add(new JsonPrimitive(nnId));
		request.add("params", params);
		
		return request;
	}
	
	public static JsonObject createForwardRequest(int id, String nnId, int[] dims){
		JsonObject request = new JsonObject();
		request.add("jsonrpc", new JsonPrimitive("2.0"));
		request.add("method", new JsonPrimitive("forward"));
		request.add("id", new JsonPrimitive(id));
		
		JsonArray params = new JsonArray();
		params.add(new JsonPrimitive(nnId));
		
		// create input
		JsonArray input = new JsonArray();
		List<JsonArray> toAdd = new ArrayList<>();
		toAdd.add(input);
		for(int i=0;i<dims.length;i++){
			List<JsonArray> nextAdd = new ArrayList<>();
			int d = dims[i];
			for(int k=0;k<d;k++){
				if(i == dims.length-1){
					// add floats
					for(JsonArray a : toAdd){
						a.add(new JsonPrimitive(0.0f));
					}
				} else {
					// add jsonarrays
					for(JsonArray a : toAdd){
						JsonArray newArray = new JsonArray();
						a.add(newArray);
						nextAdd.add(newArray);
					}
				}
			}
			toAdd = nextAdd;
		}
		params.add(input);
		
		request.add("params", params);
		
		return request;
	}
	
	private static JsonObject createJsonFromMap(Map<String, String> map){
		JsonObject json = new JsonObject();
		for(Entry<String, String> e : map.entrySet()){
			json.add(e.getKey(), new JsonPrimitive(e.getValue()));
		}
		return json;
	}
	
}
