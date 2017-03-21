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
package be.iminds.iot.dianne.jsonrpc;

import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.util.Map;
import java.util.UUID;
import java.util.stream.Collectors;

import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;
import org.osgi.util.promise.Promise;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParseException;
import com.google.gson.JsonParser;
import com.google.gson.stream.JsonReader;
import com.google.gson.stream.JsonWriter;

import be.iminds.iot.dianne.api.coordinator.AgentResult;
import be.iminds.iot.dianne.api.coordinator.DianneCoordinator;
import be.iminds.iot.dianne.api.coordinator.EvaluationResult;
import be.iminds.iot.dianne.api.coordinator.LearnResult;
import be.iminds.iot.dianne.api.dataset.DianneDatasets;
import be.iminds.iot.dianne.api.nn.Dianne;
import be.iminds.iot.dianne.api.nn.NeuralNetwork;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkDTO;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkInstanceDTO;
import be.iminds.iot.dianne.api.nn.platform.DiannePlatform;
import be.iminds.iot.dianne.coordinator.util.DianneCoordinatorWriter;
import be.iminds.iot.dianne.nn.util.DianneJSONConverter;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;

@Component
public class DianneRequestHandler implements JSONRPCRequestHandler {

	private DianneCoordinator coordinator;
	private DiannePlatform platform;
	private DianneDatasets datasets;
	private Dianne dianne;
	
	@Override
	public void handleRequest(JsonReader reader, JsonWriter writer) throws IOException {
		try {
			JsonParser parser = new JsonParser();
			JsonObject request = parser.parse(reader).getAsJsonObject();
			handleRequest(request, writer);
		} catch(JsonParseException e){
			e.printStackTrace();
			writeError(writer, null, -32700, "Parse error");
		} catch(IllegalStateException e){
			// this happens when the client closes the socket and reader returns null
			throw new IOException(e);
		}
	}
	
	@Override
	public void handleRequest(JsonObject request, JsonWriter writer) throws IOException {
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
		case "deploy":
			try {
				String description = null;
				UUID runtimeId = null;
				String[] tags = null;
				
				NeuralNetworkInstanceDTO nni;
				JsonArray params = request.get("params").getAsJsonArray();
				if(params.size() > 1){
					description = params.get(1).getAsString();
				}
				if(params.size() > 2){
					runtimeId = UUID.fromString(params.get(2).getAsString());
				}
				if(params.size() > 3){
					tags = new String[params.size()-3];
					for(int t=3;t<params.size();t++){
						tags[t-3] = params.get(t).getAsString();
					}
				}
				
				if(params.get(0).isJsonPrimitive()){
					String nnName = params.get(0).getAsString();
					nni = platform.deployNeuralNetwork(nnName, description, runtimeId, tags);
				} else {
					NeuralNetworkDTO nn = DianneJSONConverter.parseJSON(params.get(0).getAsJsonObject());
					nni = platform.deployNeuralNetwork(nn, description, runtimeId, tags);
				}
				writeResult(writer, id, nni.id.toString());
			} catch(Exception e){
				writeError(writer, id, -32602, "Incorrect parameters provided: "+e.getMessage());
				return;
			}
			break;
		case "undeploy":
			try {
				JsonArray params = request.get("params").getAsJsonArray();
				if(params.get(0).isJsonPrimitive()){
					String s = params.get(0).getAsString();
					UUID nnId = UUID.fromString(s);
					platform.undeployNeuralNetwork(nnId);
					writeResult(writer, id, nnId);
				} 
			} catch(Exception e){
				writeError(writer, id, -32602, "Incorrect parameters provided: "+e.getMessage());
				return;
			}
			break;
		case "forward":
			try {
				JsonArray params = request.get("params").getAsJsonArray();
				if(params.size() != 2){
					throw new Exception("2 parameters expected");
				}
				if(!params.get(0).isJsonPrimitive())
					throw new Exception("first parameter should be neural network instance id");
				if(!params.get(1).isJsonArray())
					throw new Exception("second parameter should be input data");
				
				String s = params.get(0).getAsString();
				UUID nnId = UUID.fromString(s);
				NeuralNetworkInstanceDTO nni = platform.getNeuralNetworkInstance(nnId);
				if(nni==null){
					writeError(writer, id, -32603, "Neural network with id "+nnId+" does not exist.");
					return;
				}
				NeuralNetwork nn = dianne.getNeuralNetwork(nni).getValue();
				
				JsonArray in = params.get(1).getAsJsonArray();
				Tensor input = asTensor(in);
				nn.forward(null, null, input).then(p -> {
					int argmax = TensorOps.argmax(p.getValue().tensor);
					writeResult(writer, id, nn.getOutputLabels()[argmax]);
					return null;
				}, p -> {
					writeError(writer, id, -32603, "Error during forward: "+p.getFailure().getMessage());
				});
				
			} catch(Exception e){
				writeError(writer, id, -32602, "Incorrect parameters provided: "+e.getMessage());
				return;
			}
			break;
		case "learn":
		case "eval":
		case "act":
			String[] nnName = null;
			NeuralNetworkDTO nn = null;
			String dataset;
			Map<String, String> config;
			
			try {
				JsonArray params = request.get("params").getAsJsonArray();
				if(params.get(0).isJsonPrimitive()){
					nnName = params.get(0).getAsString().split(",");
					for(int k=0;k<nnName.length;k++){
						nnName[k] = nnName[k].trim();
					}
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
					result= coordinator.learn(dataset, config, nnName);
				} else {
					result = coordinator.learn(dataset, config, nn);
				}
				try {
					result.then(p -> {
						writeResult(writer, id, p.getValue());
						return null;
					}, p -> {
						writeError(writer, id, -32603, "Error during learning: "+p.getFailure().getMessage());
					}).getValue();
				} catch (InvocationTargetException | InterruptedException e) {
					e.printStackTrace();
				}
			} else if(method.equals("eval")){
				// eval
				Promise<EvaluationResult> result = null;
				if(nnName!=null){
					result= coordinator.eval(dataset, config, nnName);
				} else {
					result = coordinator.eval(dataset, config, nn);
				}
				try {
					result.then(p -> {
						writeResult(writer, id, p.getValue());
						return null;
					}, p -> {
						writeError(writer, id, -32603, "Error during learning: "+p.getFailure().getMessage());
					}).getValue();
				} catch (InvocationTargetException | InterruptedException e) {
					e.printStackTrace();
				}
			} else if(method.equals("act")){
				Promise<AgentResult> result = null;
				if(nnName!=null){
					result= coordinator.act(dataset, config, nnName);
				} else {
					result = coordinator.act(dataset, config, nn);
				}
				try {
					result.then(p -> {
						writeResult(writer, id, null);
						return null;
					}, p -> {
						writeError(writer, id, -32603, "Error during learning: "+p.getFailure().getMessage());
					}).getValue();
				} catch (InvocationTargetException | InterruptedException e) {
					e.printStackTrace();
				}
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
					writeResult(writer, id, "Job "+jobId+" stopped");
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
			writeResult(writer, id, datasets.getDatasets().stream().map(d -> d.name).collect(Collectors.toList()));
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
	
	private void writeError(JsonWriter writer, String id, int code, String message) throws IOException {
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
	
	private void writeResult(JsonWriter writer, String id, Object result) throws IOException {
		writer.beginObject();
		writer.name("jsonrpc");
		writer.value("2.0");
		writer.name("id");
		writer.value(id);
		writer.name("result");
		// write result object
		try {
			if(result instanceof LearnResult){
				DianneCoordinatorWriter.writeLearnResult(writer, (LearnResult)result);
			} else if(result instanceof AgentResult){
				DianneCoordinatorWriter.writeAgentResult(writer, (AgentResult)result);
			} else {
				DianneCoordinatorWriter.writeObject(writer, result);
			}
		} catch(Throwable t){
			t.printStackTrace();
		}
		// end result object
		writer.endObject();
		writer.flush();		
	}
	
	private Tensor asTensor(JsonArray array){
		// support up to 3 dim input atm
		int dim0 = 1;
		int dim1 = 1;
		int dim2 = 1;
		
		int dims = 1;
		dim0 = array.size();
		if(array.get(0).isJsonArray()){
			dims = 2;
			JsonArray a = array.get(0).getAsJsonArray();
			dim1 = a.size();
			if(a.get(0).isJsonArray()){
				dims = 3;
				dim2 = a.get(0).getAsJsonArray().size();
			}
		}
		
		int size = dim0*dim1*dim2;
		float[] data = new float[size];
		int k = 0;
		for(int i=0;i<dim0;i++){
			for(int j=0;j<dim1;j++){
				for(int l=0;l<dim2;l++){
					JsonElement e = array.get(i);
					if(e.isJsonArray()){
						e = e.getAsJsonArray().get(j);
						if(e.isJsonArray()){
							e = e.getAsJsonArray().get(l);
						}
					}
					data[k++] = e.getAsFloat();
				}
			}
		}
		
		int[] d = new int[dims];
		d[0] = dim0;
		if(dims > 1)
			d[1] = dim1;
		if(dims > 2)
			d[2] = dim2;
		
		return new Tensor(data, d);
	}
	
	@Reference
	void setDianneCoordinator(DianneCoordinator c){
		this.coordinator = c;
	}

	@Reference
	void setDiannePlatform(DiannePlatform p){
		this.platform = p;
	}
	
	@Reference
	void setDianneDatasets(DianneDatasets d){
		this.datasets = d;
	}
	
	@Reference
	void setDianne(Dianne d){
		this.dianne = d;
	}
}