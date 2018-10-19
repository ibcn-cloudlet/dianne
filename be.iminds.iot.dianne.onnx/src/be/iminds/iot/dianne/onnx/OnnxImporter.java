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
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.stream.Collectors;

import com.google.protobuf.CodedInputStream;

import be.iminds.iot.dianne.api.nn.module.dto.ModuleDTO;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkDTO;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;
import onnx.Onnx.GraphProto;
import onnx.Onnx.ModelProto;
import onnx.Onnx.NodeProto;
import onnx.Onnx.TensorProto;
import onnx.Onnx.TensorProto.DataType;

public class OnnxImporter {

	private ModelProto model;
	
	private String name;
	private List<ModuleDTO> modules = new ArrayList<>();
	private Map<UUID, Tensor> parameters = new HashMap<>();
	private Map<String, Tensor> constants = new HashMap<>();

	public OnnxImporter(String onnxFile) {
		parse(onnxFile);
	}
	
	public NeuralNetworkDTO getNN() {
		NeuralNetworkDTO nn = new NeuralNetworkDTO(name, modules);
		return nn;
	}
	
	public NeuralNetworkDTO getNN(String name) {
		NeuralNetworkDTO nn = new NeuralNetworkDTO(name, modules);
		return nn;
	}
	
	public Map<UUID, Tensor> getParameters(){
		return parameters;
	}
	
	private void parse(String onnxFile) {
		try {
			CodedInputStream in = CodedInputStream.newInstance(new FileInputStream(onnxFile));
			in.setSizeLimit(Integer.MAX_VALUE);
			model = ModelProto.parseFrom(in);
			GraphProto graph = model.getGraph();

			name = graph.getName();
			
			// map output tensor ids to their modules
			Map<String, ModuleDTO> outputMap = new HashMap<>();
			// map weight tensors ids to their tensor value
			// TODO are these always in initializer list? not in spec...
			Map<String, TensorProto> tensorMap = new HashMap<>();
			graph.getInitializerList().forEach(init -> {
				tensorMap.put(init.getName(), init);
			});
			
			ModuleDTO input = new ModuleDTO();
			input.type = "Input";
			input.properties.put("name", "Input");
			modules.add(input);
			
			// parse nodes
			List<NodeProto> nodes = graph.getNodeList();
			nodes.forEach(node -> {
				System.out.println("Parsing "+node.getName()+" "+node.getOpType());
				node.getAttributeList().forEach(attr -> System.out.println(" * "+attr.getName()));
				
				ModuleDTO dto = new ModuleDTO();

				if(node.getOpType().equals("Constant")) {
					Tensor c = toTensor(node.getAttribute(0).getT());
					constants.put(node.getOutput(0), c);
					return;
				}
				
				// get previous
				ModuleDTO prev =  outputMap.get(node.getInput(0));
				if(prev == null) {
					prev = input;
				}
				
				// fill output map
				outputMap.put(node.getOutput(0), dto);
				
				String type = node.getOpType();
				switch(type) {
				case "Relu":
				{
					dto.type = "ReLU";
					break;
				}
				case "Sigmoid":
				{
					dto.type = "Sigmoid";
					break;
				}
				case "Tanh":
				{
					dto.type = "Tanh";
					break;
				}
				case "Elu":
				{
					dto.type = "ELU";
					
					node.getAttributeList().forEach(attr ->{
						switch(attr.getName()) {
						case "alpha":
							dto.properties.put("alpha", ""+attr.getF());								
							break;
						}
					});					
					break;
				}
				case "PRelu":
				{
					dto.type = "PReLU";
					
					Tensor params = toTensor(tensorMap.get(node.getInput(1)));
					parameters.put(dto.id, params);
					break;
				}
				case "Selu":
				{
					dto.type = "SELU";
					break;
				}
				case "Softplus":
				{
					dto.type = "Softplus";
					break;
				}
				case "Softmax":
				{
					dto.type = "Softmax";
					break;
				}
				case "LogSoftmax":
				{
					dto.type = "LogSoftmax";
					break;
				}
				case "Gemm":
				{
					dto.type = "Linear";
					
					TensorProto w = tensorMap.get(node.getInput(1));
					TensorProto b = null;
					if(node.getInputCount() > 2) {
						b = tensorMap.get(node.getInput(2));
					}
					
					int oo = (int) w.getDims(0);
					int ii = (int) w.getDims(1);
					dto.properties.put("output", ""+oo);
					dto.properties.put("input", ""+ii);

					// TODO process attributes
//					node.getAttributeList().forEach(attr ->{
//						switch(attr.getName()) {
//						case "alpha":
//							break;
//						case "beta":
//							break;
//						case "broadcast":
//							break;
//						case "transA":
//							break;
//						case "transB":
//							break;
//						}
//					});
					
					// create params Tensor
					Tensor params = new Tensor(oo*(ii+1));
					params.fill(0.0f);
					toTensor(w).copyInto(params.narrow(0, 0, oo*ii));
					if(b != null) {
						toTensor(b).copyInto(params.narrow(0, oo*ii, oo));
					}
					
					parameters.put(dto.id, params);
					
					break;
				}
				case "Conv":
				{
					dto.type = "Convolution";
	
					// get attributes
					node.getAttributeList().forEach(attr ->{
						switch(attr.getName()) {
						case "auto_pad":
							throw new RuntimeException("auto_pad not supported, use the pads attribute");
						case "dilatations":
							throw new RuntimeException("Dianne has no dilatation support atm");
						case "group":
							int group = (int)attr.getI();
							if(group != 1)
								throw new RuntimeException("Dianne has no group support atm");
							break;
						case "kernel_shape":
							List<Long> shape = attr.getIntsList();

							dto.properties.put("kernelWidth", ""+shape.get(0));
							if(shape.size() > 1) {
								dto.properties.put("kernelHeight", ""+shape.get(1));
							}
							if(shape.size() > 2) {
								dto.properties.put("kernelDepth", ""+shape.get(2));
							}
							if(shape.size() > 3) {
								throw new RuntimeException("Dianne doesn't support kernel sizes larger than 3d");
							}
							break;
						case "pads":
							List<Long> pads = attr.getIntsList();
							if(pads.size()>1) {
								if(pads.get(0) != pads.get(1)) {
									throw new RuntimeException("Dianne only support symmetric pads atm");
								} else {
									dto.properties.put("padX", ""+pads.get(0));
								}
							}
							if(pads.size()>3) {
								if(pads.get(2) != pads.get(3)) {
									throw new RuntimeException("Dianne only support symmetric pads atm");
								} else {
									dto.properties.put("padY", ""+pads.get(2));
								}
							}
							if(pads.size()>5) {
								if(pads.get(4) != pads.get(5)) {
									throw new RuntimeException("Dianne only support symmetric pads atm");
								} else {
									dto.properties.put("padZ", ""+pads.get(4));
								}
							}
							if(pads.size() > 6) {
								throw new RuntimeException("Dianne doesn't support kernel sizes larger than 3d");
							}
							break;
						case "strides":
							List<Long> stride = attr.getIntsList();
							if(stride.size() > 0) {
								dto.properties.put("strideX", ""+stride.get(0));
							}
							if(stride.size() > 1) {
								dto.properties.put("strideY", ""+stride.get(1));
							}
							if(stride.size() > 2) {
								dto.properties.put("strideZ", ""+stride.get(2));
							}
							if(stride.size() > 3) {
								throw new RuntimeException("Dianne doesn't support stride sizes larger than 3d");
							}
							break;
						}
					});
					
					// get weights
					TensorProto w = tensorMap.get(node.getInput(1));
					int noInputPlanes = (int)w.getDims(1);
					int noOutputPlanes = (int)w.getDims(0);
					dto.properties.put("noInputPlanes", ""+noInputPlanes);
					dto.properties.put("noOutputPlanes", ""+noOutputPlanes);
					
					TensorProto b = null;
					if(node.getInputCount() == 3) {
						b = tensorMap.get(node.getInput(2));
					}

					// create params Tensor
					Tensor tw = toTensor(w);
					Tensor params = new Tensor(tw.size()+noOutputPlanes);
					params.fill(0.0f);
					tw.copyInto(params.narrow(0, 0, tw.size()));
					if(b != null) {
						toTensor(b).copyInto(params.narrow(0, tw.size(), noOutputPlanes));
					}
					parameters.put(dto.id, params);
					
					break;
				}
				case "MaxPool":
				{
					dto.type = "MaxPooling";
					
					// get attributes
					node.getAttributeList().forEach(attr ->{
						switch(attr.getName()) {
						case "auto_pad":
							throw new RuntimeException("auto_pad not supported, use the pads attribute");
						case "kernel_shape":
							List<Long> shape = attr.getIntsList();
							dto.properties.put("width", ""+shape.get(0));
							if(shape.size() > 1) {
								dto.properties.put("height", ""+shape.get(1));
							}
							if(shape.size() > 2) {
								dto.properties.put("depth", ""+shape.get(2));
							}
							if(shape.size() > 3) {
								throw new RuntimeException("Dianne doesn't support kernel sizes larger than 3d");
							}
							break;
						case "pads":
							List<Long> pads = attr.getIntsList();
							if(pads.size()>1) {
								if(pads.get(0) != pads.get(1)) {
									throw new RuntimeException("Dianne only support symmetric pads atm");
								} else {
									dto.properties.put("padX", ""+pads.get(0));
								}
							}
							if(pads.size()>3) {
								if(pads.get(2) != pads.get(3)) {
									throw new RuntimeException("Dianne only support symmetric pads atm");
								} else {
									dto.properties.put("padY", ""+pads.get(2));
								}
							}
							if(pads.size()>5) {
								if(pads.get(4) != pads.get(5)) {
									throw new RuntimeException("Dianne only support symmetric pads atm");
								} else {
									dto.properties.put("padZ", ""+pads.get(4));
								}
							}
							if(pads.size() > 6) {
								throw new RuntimeException("Dianne doesn't support kernel sizes larger than 3d");
							}
							break;
						case "strides":
							List<Long> stride = attr.getIntsList();
							if(stride.size() > 0) {
								dto.properties.put("strideX", ""+stride.get(0));
							}
							if(stride.size() > 1) {
								dto.properties.put("strideY", ""+stride.get(1));
							}
							if(stride.size() > 2) {
								dto.properties.put("strideZ", ""+stride.get(2));
							}
							if(stride.size() > 3) {
								throw new RuntimeException("Dianne doesn't support stride sizes larger than 3d");
							}
							break;
						}
					});
					
					break;
				}
				case "AveragePool":
				{
					dto.type = "AvgPooling";
					
					// get attributes
					node.getAttributeList().forEach(attr ->{
						switch(attr.getName()) {
						case "auto_pad":
							throw new RuntimeException("auto_pad not supported, use the pads attribute");
						case "kernel_shape":
							List<Long> shape = attr.getIntsList();
							dto.properties.put("width", ""+shape.get(0));
							if(shape.size() > 1) {
								dto.properties.put("height", ""+shape.get(1));
							}
							if(shape.size() > 2) {
								dto.properties.put("depth", ""+shape.get(2));
							}
							if(shape.size() > 3) {
								throw new RuntimeException("Dianne doesn't support kernel sizes larger than 3d");
							}
							break;
						case "pads":
							List<Long> pads = attr.getIntsList();
							if(pads.size()>1) {
								if(pads.get(0) != pads.get(1)) {
									throw new RuntimeException("Dianne only support symmetric pads atm");
								} else {
									dto.properties.put("padX", ""+pads.get(0));
								}
							}
							if(pads.size()>3) {
								if(pads.get(2) != pads.get(3)) {
									throw new RuntimeException("Dianne only support symmetric pads atm");
								} else {
									dto.properties.put("padY", ""+pads.get(2));
								}
							}
							if(pads.size()>5) {
								if(pads.get(4) != pads.get(5)) {
									throw new RuntimeException("Dianne only support symmetric pads atm");
								} else {
									dto.properties.put("padZ", ""+pads.get(4));
								}
							}
							if(pads.size() > 6) {
								throw new RuntimeException("Dianne doesn't support kernel sizes larger than 3d");
							}
							break;
						case "strides":
							List<Long> stride = attr.getIntsList();
							if(stride.size() > 0) {
								dto.properties.put("strideX", ""+stride.get(0));
							}
							if(stride.size() > 1) {
								dto.properties.put("strideY", ""+stride.get(1));
							}
							if(stride.size() > 2) {
								dto.properties.put("strideZ", ""+stride.get(2));
							}
							if(stride.size() > 3) {
								throw new RuntimeException("Dianne doesn't support stride sizes larger than 3d");
							}
							break;
						}
					});
					
					break;
				}
				case "LRN":
				{
					dto.type = "LRN";
					
					node.getAttributeList().forEach(attr ->{
						switch(attr.getName()) {
						case "size":
							dto.properties.put("size", ""+attr.getI());
							break;
						case "alpha":
							dto.properties.put("alpha", ""+attr.getF());
							break;
						case "beta":
							dto.properties.put("beta", ""+attr.getF());
							break;
						case "bias":
							dto.properties.put("k", ""+attr.getF());
							break;
						}
					});
					break;
				}
				case "BatchNormalization":
				{
					dto.type = "BatchNormalization";
					
					Tensor w = toTensor(tensorMap.get(node.getInput(1)));
					Tensor b = toTensor(tensorMap.get(node.getInput(2)));
					Tensor rm = toTensor(tensorMap.get(node.getInput(3)));
					Tensor rv = toTensor(tensorMap.get(node.getInput(4)));
					
					int size = w.size();
					dto.properties.put("size", ""+size);
					
					Tensor params = new Tensor(4*size);
					w.copyInto(params.narrow(0, 0, size));
					b.copyInto(params.narrow(0, size, size));
					rm.copyInto(params.narrow(0, 2*size, size));
					rv.copyInto(params.narrow(0, 3*size, size));
					
					parameters.put(dto.id, params);
					break;
				}
				case "Reshape":
				{
					dto.type = "Reshape";
					
					node.getAttributeList().forEach(attr ->{
						switch(attr.getName()) {
						case "shape":
							List<Long> shape = attr.getIntsList();
							if(shape.get(0) == -1) {
								// in case -1 is used to indicate batch dimension, omit it
								// Dianne automatically detects this...
								for(int i=1;i<shape.size();i++) {
									dto.properties.put("dim"+(i-1), ""+shape.get(i));								
								}
							} else {
								for(int i=0;i<shape.size();i++) {
									dto.properties.put("dim"+i, ""+shape.get(i));								
								}
							}
							break;
						}
					});
					
					if(node.getInputCount() == 2) {
						// shape is provided as constant input
						String c = node.getInput(1);
						Tensor shape = constants.get(c);
						if(shape.get(0) == -1) {
							// in case -1 is used to indicate batch dimension, omit it
							// Dianne automatically detects this...
							for(int i=1;i<shape.size();i++) {
								dto.properties.put("dim"+(i-1), ""+(int)shape.get(i));								
							}
						} else {
							for(int i=0;i<shape.size();i++) {
								dto.properties.put("dim"+i, ""+(int)shape.get(i));								
							}
						}
					}
					
					break;
				}
				case "Dropout":
				{
					dto.type = "Dropout";
					
					node.getAttributeList().forEach(attr ->{
						switch(attr.getName()) {
						case "ratio":
							dto.properties.put("rate", ""+attr.getF());
							break;
						}
					});					
					break;
				}
				case "Sum": {
					dto.type = "Accumulate";
					
					// this one has multiple previous modules
					ModuleDTO[] prevs = new ModuleDTO[node.getInputCount()];
					for(int i=0;i<prevs.length;i++) {
						prevs[i] =  outputMap.get(node.getInput(i));
					}
					setPrev(dto, prevs);
					break;
				}
				case "Concat": {
					dto.type = "Concat";
					
					// this one has multiple previous modules
					ModuleDTO[] prevs = new ModuleDTO[node.getInputCount()];
					for(int i=0;i<prevs.length;i++) {
						prevs[i] =  outputMap.get(node.getInput(i));
					}
					setPrev(dto, prevs);
					break;
				}
				case "Add": {
					// Atm we don't have a separate Add module
					// However, torch exports the Conv bias as a separate Add,
					// but in our case this is part of the Convolution
					// so we can just add this to the bias of the previous
					
					if(prev.type.equals("Convolution")) {
						Tensor b = toTensor(tensorMap.get(node.getInput(1)));

						// TODO add bias
						Tensor params = parameters.get(prev.id);
						Tensor bias = params.narrow(0, params.size()-b.size(), b.size());
						TensorOps.add(bias, bias, b);
						
						outputMap.put(node.getOutput(0), prev);
						return;
					} else {
						throw new RuntimeException("ONNX node type Add only supported after Conv as bias!");
					}
				}
				default:
					System.err.println("Unsupported ONNX Operator: "+node.getOpType());
					System.err.println(node.getName());
					System.err.println("inputs:");
					node.getInputList().forEach(i -> System.err.println("* "+i));
					System.err.println("outputs:");
					node.getOutputList().forEach(o -> System.err.println("* "+o));
					System.err.println("attributes:");
					node.getAttributeList().forEach(a -> {
						System.err.println(a.getName()+" : "+a.getType());
						a.getIntsList().forEach(i -> System.err.println(i));
					});
					throw new RuntimeException("ONNX node type "+type+" not supported!");
				}
				
				// configure prev/next if not done already
				if(dto.prev == null)
					setPrev(dto, prev);
				
				String n = node.getName();
				if(n == null || n.isEmpty()) {
					n  = dto.type;
				}
				dto.properties.put("name", n);
				modules.add(dto);
			});
			
			
			// now add output modules to all modules without next
			modules.addAll(modules.stream().filter(m -> m.next==null).map(m -> {
				ModuleDTO output = new ModuleDTO();
				output.type = "Output";
				output.properties.put("name", "Output");
				output.prev = new UUID[] {m.id};
				m.next = new UUID[] {output.id};
				return output;
			}).collect(Collectors.toList()));
			
		} catch(Exception e) {
			e.printStackTrace();
			throw new RuntimeException("Failed to import ONNX file "+onnxFile, e);
		}
		
	}
	
	private void setPrev(ModuleDTO dto, ModuleDTO... prevs) {
		// set prev
		UUID[] prevIds = new UUID[prevs.length];
		for(int i=0;i<prevs.length;i++) {
			prevIds[i] = prevs[i].id;
		}
		dto.prev = prevIds;
		
		for(ModuleDTO prev : prevs) {
			if(prev.next == null) {
				prev.next = new UUID[] {dto.id};
			} else {
				// add current dto to this prev's next array
				UUID[] nxt = new UUID[prev.next.length + 1];
				for(int i=0;i<prev.next.length;i++) {
					nxt[i] = prev.next[i];
				}
				nxt[prev.next.length] = dto.id;
				
				if(prev.next.length == 1) {
					// in this case onnx implicitly duplicates, in Dianne this is explicit
					ModuleDTO duplicate = new ModuleDTO();
					duplicate.type = "Duplicate";
					duplicate.properties.put("name", "Duplicate");
					
					// connect duplicate to prev
					duplicate.prev = new UUID[] {prev.id};
					prev.next = new UUID[] {duplicate.id};
	
					// next of duplicate is combination of the two
					duplicate.next = nxt;
					
					// this one's previous is the duplicate instead of prev
					for(int i=0;i<dto.prev.length;i++) {
						if(dto.prev[i].equals(prev.id)) {
							dto.prev[i] = duplicate.id;
						}
					}
					
					// the other one's previous should be changed to duplicate
					ModuleDTO o = modules.stream().filter(m -> nxt[0].equals(m.id)).findFirst().get();
					for(int k=0;k<o.prev.length;k++) {
						if(o.prev[k].equals(nxt[0])) {
							o.prev[k] = duplicate.id;
						}
					}
					
					modules.add(duplicate);
				} else { 					
					prev.next = nxt;
				}
			}
		}
	}
	
	private Tensor toTensor(TensorProto t) {
		int[] dims = new int[t.getDimsCount()];
		for(int i=0;i<dims.length;i++) {
			dims[i] = (int)t.getDims(i);
		}

		float[] data;
		ByteBuffer buffer = t.getRawData().asReadOnlyByteBuffer().order(ByteOrder.LITTLE_ENDIAN);
		if(t.getDataType() == DataType.FLOAT) {
			data = new float[buffer.capacity()/4];
			for(int i=0;i<data.length;i++) {
				data[i] = buffer.getFloat();
			}
		} else if(t.getDataType() == DataType.INT64) {
			data = new float[buffer.capacity()/8];
			for(int i=0;i<data.length;i++) {
				data[i] = (float)buffer.getLong();
			}
		} else if(t.getDataType() == DataType.INT32) {
			data = new float[buffer.capacity()/4];
			for(int i=0;i<data.length;i++) {
				data[i] = (float)buffer.getInt();
			}
		} else {
			throw new RuntimeException("Unsupported data format "+t.getDataType().toString());
		}
		
		
		return new Tensor(data, dims);
	}
}
