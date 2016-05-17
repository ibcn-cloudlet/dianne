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

import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.Socket;
import java.util.HashMap;
import java.util.Map;

import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.google.gson.stream.JsonReader;

import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkDTO;
import be.iminds.iot.dianne.nn.util.DianneJSONRPCRequestFactory;
import be.iminds.iot.dianne.nn.util.DianneNeuralNetworkBuilder;
import be.iminds.iot.dianne.nn.util.DianneNeuralNetworkBuilder.Activation;

public class JSONRPCTester {

	// TODO convert this into a proper junit test?!
	public static void main(String[] args) throws Exception {
		
		try (Socket s = new Socket("127.0.0.1", 9090)) {
			
			OutputStream out = s.getOutputStream();
			out.flush();
			JsonReader reader = new JsonReader(new InputStreamReader(s.getInputStream()));
			JsonParser parser = new JsonParser();

			// learn
			// byte[] learnRequest =
			// Files.readAllBytes(FileSystems.getDefault().getPath("test",
			// "learnRequest"));
			NeuralNetworkDTO nn = DianneNeuralNetworkBuilder.createMLP("Test", 784, 10, Activation.Sigmoid, 20);
			String dataset = "MNIST";
			Map<String, String> learnProperties = new HashMap<>();
			learnProperties.put("clean", "true");
			learnProperties.put("trace", "true");
			learnProperties.put("tag", "test");
			learnProperties.put("learningRate", "0.01");
			learnProperties.put("batchSize", "10");
			learnProperties.put("momentum", "0.9");
			learnProperties.put("regularization", "0.0001");
			learnProperties.put("trainingSet", "0:60000");
			learnProperties.put("maxIterations", "10000");
			JsonObject learnRequestJson = DianneJSONRPCRequestFactory.createLearnRequest(1, nn, dataset,
					learnProperties);
			byte[] learnRequest = learnRequestJson.toString().getBytes();
			out.write(learnRequest);

			JsonElement learnResult = parser.parse(reader);
			System.out.println(learnResult.toString());

			// eval
			// byte[] evalRequest =
			// Files.readAllBytes(FileSystems.getDefault().getPath("test",
			// "evalRequest"));
			Map<String, String> evalProperties = new HashMap<>();
			evalProperties.put("clean", "true");
			evalProperties.put("tag", "test");
			evalProperties.put("testSet", "60000:70000");
			JsonObject evalRequestJson = DianneJSONRPCRequestFactory.createEvalRequest(2, nn.name, dataset,
					evalProperties);
			byte[] evalRequest = evalRequestJson.toString().getBytes();
			out.write(evalRequest);

			JsonElement evalResult = parser.parse(reader);
			String result = evalResult.toString();
			if (result.length() > 1000) {
				result = result.substring(0, 1000) + "...";
			}
			System.out.println(result);
			
			// forward
			JsonObject deployRequestJson = DianneJSONRPCRequestFactory.createDeployRequest(3, nn.name);
			byte[] deployRequest = deployRequestJson.toString().getBytes();
			out.write(deployRequest);
			JsonElement deployResult = parser.parse(reader);
			String nnId = deployResult.getAsJsonObject().get("result").getAsString();
			System.out.println("Deployed "+nnId);
			
			JsonObject forwardRequestJson = DianneJSONRPCRequestFactory.createForwardRequest(4, nnId, new int[]{1,28,28});
			byte[] forwardRequest = forwardRequestJson.toString().getBytes();
			out.write(forwardRequest);
			JsonElement forwardResult = parser.parse(reader);
			String classification = forwardResult.getAsJsonObject().get("result").getAsString();
			System.out.println("Result "+classification);
			
			JsonObject undeployRequestJson = DianneJSONRPCRequestFactory.createUndeployRequest(5, nnId);
			byte[] undeployRequest = undeployRequestJson.toString().getBytes();
			out.write(undeployRequest);
			JsonElement undeployResult = parser.parse(reader);
			System.out.println("Undeployed "+deployResult.getAsJsonObject().get("result").getAsString());
		}
	}
}
