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
package be.iminds.iot.dianne.onnx;

import java.io.FileInputStream;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.Map;
import java.util.Queue;
import java.util.Set;
import java.util.UUID;

import com.google.protobuf.CodedInputStream;

import be.iminds.iot.dianne.api.nn.module.dto.ModuleDTO;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkDTO;
import be.iminds.iot.dianne.tensor.Tensor;
import onnx.Onnx.GraphProto;
import onnx.Onnx.ModelProto;
import onnx.Onnx.NodeProto;
import onnx.Onnx.TensorProto;

public class OnnxLoader {

	private Map<UUID, Tensor> parameters = new HashMap<>();

	public OnnxLoader(String onnxFile, NeuralNetworkDTO nn) {
		parse(onnxFile, nn);
	}
	
	public Map<UUID, Tensor> getParameters(){
		return parameters;
	}
	
	private void parse(String onnxFile, NeuralNetworkDTO network) {
		try {
			CodedInputStream in = CodedInputStream.newInstance(new FileInputStream(onnxFile));
			in.setSizeLimit(Integer.MAX_VALUE);
			ModelProto model = ModelProto.parseFrom(in);
			GraphProto graph = model.getGraph();
			Iterator<NodeProto> nodeIterator = graph.getNodeList().iterator();
			// map weight tensors ids to their tensor value
			// TODO are these always in initializer list? not in spec...
			Map<String, TensorProto> tensorMap = new HashMap<>();
			graph.getInitializerList().forEach(init -> {
				tensorMap.put(init.getName(), init);
			});
			
			// use linked list to start from input module
			Queue<ModuleDTO> q = new LinkedList<>();
			for (ModuleDTO module : network.modules.values()) {
				if (module.type.equals("Input")) {
					q.add(module);
				}
			}
			if (q.isEmpty())
				throw new RuntimeException("No input module for this network!");
			
			// iterate over modules and find matching onnx modules at the same time
			Set<UUID> visited = new HashSet<>();
			while (!q.isEmpty()) {
				ModuleDTO module = q.poll();
				if (visited.contains(module.id))
					continue;
				visited.add(module.id);
				
				UUID[] nextIds = module.next;
				if (nextIds != null) {
					for (UUID nextId : nextIds) {
						q.add(network.modules.get(nextId));
					}
				}
				
				String onnxType = null;
				switch(module.type) {
				case "Convolution":
					onnxType = "Conv";
					break;
				case "Linear":
					onnxType = "Gemm";
					break;
				default:
					System.out.println(module.type + " parameter load has not been implemented. Does it use weights?");
					continue;
				}
				
				// move the onnx graph and search for matching node
				NodeProto matchedNode = null;
				while (nodeIterator.hasNext()) {
					NodeProto node = nodeIterator.next();
					if (node.getOpType().equals(onnxType)) {
						matchedNode = node;
						break;
					}
				}
				
				if (matchedNode != null) {
					switch(matchedNode.getOpType()) {
					case "Gemm":
					{
						TensorProto w = tensorMap.get(matchedNode.getInput(1));
						TensorProto b = null;
						if(matchedNode.getInputCount() > 2) {
							b = tensorMap.get(matchedNode.getInput(2));
						}
						
						int oo = (int) w.getDims(0);
						int ii = (int) w.getDims(1);
						
						// create params Tensor
						Tensor params = new Tensor(oo*(ii+1));
						params.fill(0.0f);
						OnnxUtil.toTensor(w).copyInto(params.narrow(0, 0, oo*ii));
						if(b != null) {
							OnnxUtil.toTensor(b).copyInto(params.narrow(0, oo*ii, oo));
						}
						
						parameters.put(module.id, params);
						break;
					}
					case "Conv":
					{
						// get weights
						TensorProto w = tensorMap.get(matchedNode.getInput(1));
						int noOutputPlanes = (int)w.getDims(0);
						
						TensorProto b = null;
						if(matchedNode.getInputCount() > 2) {
							b = tensorMap.get(matchedNode.getInput(2));
						}

						// create params Tensor
						Tensor tw = OnnxUtil.toTensor(w);
						Tensor params = new Tensor(tw.size()+noOutputPlanes);
						params.fill(0.0f);
						tw.copyInto(params.narrow(0, 0, tw.size()));
						if(b != null) {
							OnnxUtil.toTensor(b).copyInto(params.narrow(0, tw.size(), noOutputPlanes));
						}
						parameters.put(module.id, params);
						break;
					}
					default:
						throw new RuntimeException("This should never happen!!! Module type: " + module.type + ", OnnxType: " + onnxType + ", matchedNode type: " + matchedNode.getOpType());
					}						
				} else {
					throw new RuntimeException("no matching module found in onnx: " + onnxType);
				}
			}
			
		} catch(Exception e) {
			throw new RuntimeException("Failed to import ONNX file "+onnxFile, e);
		}
		
	}
}
