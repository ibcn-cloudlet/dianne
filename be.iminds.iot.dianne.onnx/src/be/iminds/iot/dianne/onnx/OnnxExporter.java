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

import java.io.ByteArrayOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;

import com.google.protobuf.ByteString;

import be.iminds.iot.dianne.api.nn.module.dto.ModuleDTO;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkDTO;
import be.iminds.iot.dianne.tensor.Tensor;
import onnx.Onnx.AttributeProto;
import onnx.Onnx.AttributeProto.AttributeType;
import onnx.Onnx.GraphProto;
import onnx.Onnx.ModelProto;
import onnx.Onnx.NodeProto;
import onnx.Onnx.TensorProto;
import onnx.Onnx.TensorProto.DataType;

public class OnnxExporter {

	private ModelProto.Builder model;
	private GraphProto.Builder graph;
	
	private NeuralNetworkDTO nn;
	private Map<UUID, Tensor> parameters;
	
	private Set<UUID> processed = new HashSet<>();
	
	public OnnxExporter(NeuralNetworkDTO nn, Map<UUID, Tensor> parameters) {
		this.nn = nn;
		this.parameters = parameters;
		
		this.graph = GraphProto.newBuilder()
				.setName(nn.name);
			
		nn.modules.values().stream()
			.filter(m -> m.type.equals("Input"))
			.forEach(dto -> convertModule(dto));
		
		this.model = ModelProto.newBuilder()
				.setIrVersion(2)
				.setProducerName("Dianne")
				.setGraph(graph);
	}
	
	public void export(String onnxFile) {
		try (FileOutputStream output = new FileOutputStream(onnxFile)){
			model.build().writeTo(output);
		} catch(IOException e) {
			throw new RuntimeException("Failed to write onnx file", e);
		}
	}
	
	private void convertModule(ModuleDTO dto) {
		if(processed.contains(dto.id))
			return;
		
		switch(dto.type) {
		case "Linear":
		{
			int input = Integer.parseInt(dto.properties.get("input"));
			int output = Integer.parseInt(dto.properties.get("output"));
			
			graph.addNode(NodeProto.newBuilder()
				.setOpType("Gemm")
				.addAttribute(AttributeProto.newBuilder() 
						.setName("alpha")
						.setType(AttributeType.FLOAT)
						.setF(1.0f))
				.addAttribute(AttributeProto.newBuilder()
						.setType(AttributeType.FLOAT)
						.setName("beta").setF(1.0f))
				.addAttribute(AttributeProto.newBuilder() 
						.setName("broadcast")
						.setType(AttributeType.INT)
						.setI(1))
				.addAttribute(AttributeProto.newBuilder() 
						.setName("transA")
						.setType(AttributeType.INT)
						.setI(0))
				.addAttribute(AttributeProto.newBuilder() 
						.setName("transB")
						.setType(AttributeType.INT)
						.setI(0))
				.addInput(dto.prev[0].toString())
				.addInput(dto.id.toString()+"_W")
				.addInput(dto.id.toString()+"_b")
				.addOutput(dto.id.toString())
			);
			
			Tensor params = parameters.get(dto.id);
			Tensor w = params.narrow(0, 0, output*input);
			w.reshape(output, input);
			convertTensor(dto.id.toString()+"_W", w);
			
			Tensor b = params.narrow(0, output*input, output);
			b.reshape(output);
			convertTensor(dto.id.toString()+"_b", b);
			
			break;
		}
		case "BatchNormalization":
		{
			int size = Integer.parseInt(dto.properties.get("size"));
			
			graph.addNode(NodeProto.newBuilder()
				.setOpType("BatchNormalization")
				.addInput(dto.prev[0].toString())
				.addInput(dto.id.toString()+"_W")
				.addInput(dto.id.toString()+"_b")
				.addInput(dto.id.toString()+"_mean")
				.addInput(dto.id.toString()+"_var")
				.addOutput(dto.id.toString())
			);
			
			Tensor params = parameters.get(dto.id);

			Tensor weights = params.narrow(0, 0, size);
			convertTensor(dto.id.toString()+"_W", weights);

			Tensor bias = params.narrow(0, size, size);
			convertTensor(dto.id.toString()+"_b", bias);

			Tensor rMean = params.narrow(0, 2*size, size);
			convertTensor(dto.id.toString()+"_mean", rMean);

			Tensor rVar = params.narrow(0, 3*size, size);
			convertTensor(dto.id.toString()+"_var", rVar);
			
			break;
		}
		case "Dropout":
		{
			float rate = Float.parseFloat(dto.properties.get("rate"));
			
			graph.addNode(NodeProto.newBuilder()
				.setOpType("Dropout")
				.addAttribute(AttributeProto.newBuilder() 
						.setName("ratio")
						.setType(AttributeType.FLOAT)
						.setF(rate))
				.addInput(dto.prev[0].toString())
				.addOutput(dto.id.toString())
			);
			
			break;
		}
		case "Tanh":
		{
			graph.addNode(NodeProto.newBuilder()
				.setOpType("Tanh")
				.addInput(dto.prev[0].toString())
				.addOutput(dto.id.toString())
			);
			break;
		}
		case "Sigmoid":
		{
			graph.addNode(NodeProto.newBuilder()
				.setOpType("Sigmoid")
				.addInput(dto.prev[0].toString())
				.addOutput(dto.id.toString())
			);
			break;
		}
		case "SoftPlus":
		{
			graph.addNode(NodeProto.newBuilder()
				.setOpType("Softplus")
				.addInput(dto.prev[0].toString())
				.addOutput(dto.id.toString())
			);
			break;
		}
		case "ELU":
		{
			float alpha = hasProperty(dto.properties, "alpha") ? Float.parseFloat(dto.properties.get("alpha")) : 1.0f;
			
			graph.addNode(NodeProto.newBuilder()
				.setOpType("Elu")
				.addAttribute(AttributeProto.newBuilder() 
						.setName("alpha")
						.setType(AttributeType.FLOAT)
						.setF(alpha))
				.addInput(dto.prev[0].toString())
				.addOutput(dto.id.toString())
			);			
			break;
		}
		case "Softmax":
		{
			graph.addNode(NodeProto.newBuilder()
				.setOpType("Softmax")
				.addInput(dto.prev[0].toString())
				.addOutput(dto.id.toString())
			);
			break;
		}
		case "LogSoftmax":
		{
			graph.addNode(NodeProto.newBuilder()
				.setOpType("LogSoftmax")
				.addInput(dto.prev[0].toString())
				.addOutput(dto.id.toString())
			);
			break;
		}
		case "ReLU":
		{
			graph.addNode(NodeProto.newBuilder()
				.setOpType("Relu")
				.addInput(dto.prev[0].toString())
				.addOutput(dto.id.toString())
			);
			break;
		}
		case "SELU":
		{
			graph.addNode(NodeProto.newBuilder()
				.setOpType("Selu")
				.addInput(dto.prev[0].toString())
				.addOutput(dto.id.toString())
			);
			break;
		}
		case "PReLU":
		{
			graph.addNode(NodeProto.newBuilder()
				.setOpType("Selu")
				.addInput(dto.prev[0].toString())
				.addInput(dto.prev[0].toString()+"_W")
				.addOutput(dto.id.toString())
			);
			
			Tensor weights = parameters.get(dto.id);
			convertTensor(dto.id.toString()+"_W", weights);
			break;
		}
		case "Input":
		{
			// nothing to be done
			break;
		}
		case "Output":
		{
			// nothing to be done
			break;
		}
		case "Convolution":
		{
			int noInputPlanes = Integer.parseInt(dto.properties.get("noInputPlanes"));
			int noOutputPlanes = Integer.parseInt(dto.properties.get("noOutputPlanes"));
			int kernelWidth = Integer.parseInt(dto.properties.get("kernelWidth"));
			int kernelHeight = hasProperty(dto.properties,"kernelHeight") ? Integer.parseInt(dto.properties.get("kernelHeight")) : 1;
			int kernelDepth = hasProperty(dto.properties,"kernelDepth") ? Integer.parseInt(dto.properties.get("kernelDepth")) : 1;
		
			
			List<Long> kernel = new ArrayList<>();
			kernel.add(Long.parseLong(dto.properties.get("kernelWidth")));
			if(hasProperty(dto.properties,"kernelHeight")) {
				kernel.add(Long.parseLong(dto.properties.get("kernelHeight")));
			}
			if(hasProperty(dto.properties,"kernelDepth")) {
				kernel.add(Long.parseLong(dto.properties.get("kernelDepth")));
			}
			
			List<Long> strides = new ArrayList<>();
			if(hasProperty(dto.properties,"strideX")) {
				strides.add(Long.parseLong(dto.properties.get("strideX")));
			}
			if(hasProperty(dto.properties,"strideY")) {
				strides.add(Long.parseLong(dto.properties.get("strideY")));
			}
			if(hasProperty(dto.properties,"strideZ")) {
				strides.add(Long.parseLong(dto.properties.get("strideZ")));
			}
			
			List<Long> pads = new ArrayList<>();
			if(hasProperty(dto.properties,"padX")) {
				pads.add(Long.parseLong(dto.properties.get("padX")));
				pads.add(Long.parseLong(dto.properties.get("padX")));
			}
			if(hasProperty(dto.properties,"padY")) {
				pads.add(Long.parseLong(dto.properties.get("padY")));
				pads.add(Long.parseLong(dto.properties.get("padY")));
			}
			if(hasProperty(dto.properties,"padZ")) {
				pads.add(Long.parseLong(dto.properties.get("padZ")));
				pads.add(Long.parseLong(dto.properties.get("padZ")));
			}
			
			graph.addNode(NodeProto.newBuilder()
				.setOpType("Conv")
				.addAttribute(AttributeProto.newBuilder() 
						.setName("kernel_shape")
						.setType(AttributeType.INTS)
						.addAllInts(kernel))
				.addAttribute(AttributeProto.newBuilder() 
						.setName("strides")
						.setType(AttributeType.INTS)
						.addAllInts(strides))
				.addAttribute(AttributeProto.newBuilder() 
						.setName("pads")
						.setType(AttributeType.INTS)
						.addAllInts(pads))				
				.addInput(dto.prev[0].toString())
				.addInput(dto.id.toString()+"_W")
				.addInput(dto.id.toString()+"_b")
				.addOutput(dto.id.toString())
			);
			
			Tensor params = parameters.get(dto.id);
			Tensor w = params.narrow(0, 0, noOutputPlanes*noInputPlanes*kernelWidth*kernelHeight*kernelDepth);
			int[] sizeW = new int[kernel.size()+2];
			sizeW[0] = noOutputPlanes;
			sizeW[1] = noInputPlanes;
			for(int i=0;i<kernel.size();i++) {
				sizeW[i+2] = (int)((long)kernel.get(i));
			}
			w.reshape(sizeW);
			convertTensor(dto.id.toString()+"_W", w);
			
			Tensor b = params.narrow(0, noOutputPlanes*noInputPlanes*kernelWidth*kernelHeight*kernelDepth, noOutputPlanes);
			b.reshape(noOutputPlanes);
			convertTensor(dto.id.toString()+"_b", b);
			
			break;
		}
		case "MaxPooling":
		{
			List<Long> kernel = new ArrayList<>();
			kernel.add(Long.parseLong(dto.properties.get("width")));
			if(hasProperty(dto.properties,"height")) {
				kernel.add(Long.parseLong(dto.properties.get("height")));
			}
			if(hasProperty(dto.properties,"depth")) {
				kernel.add(Long.parseLong(dto.properties.get("depth")));
			}
			
			List<Long> strides = new ArrayList<>();
			if(hasProperty(dto.properties,"strideX")) {
				strides.add(Long.parseLong(dto.properties.get("strideX")));
			}
			if(hasProperty(dto.properties,"strideY")) {
				strides.add(Long.parseLong(dto.properties.get("strideY")));
			}
			if(hasProperty(dto.properties,"strideZ")) {
				strides.add(Long.parseLong(dto.properties.get("strideZ")));
			}
			
			List<Long> pads = new ArrayList<>();
			if(hasProperty(dto.properties,"padX")) {
				pads.add(Long.parseLong(dto.properties.get("padX")));
				pads.add(Long.parseLong(dto.properties.get("padX")));
			}
			if(hasProperty(dto.properties,"padY")) {
				pads.add(Long.parseLong(dto.properties.get("padY")));
				pads.add(Long.parseLong(dto.properties.get("padY")));
			}
			if(hasProperty(dto.properties,"padZ")) {
				pads.add(Long.parseLong(dto.properties.get("padZ")));
				pads.add(Long.parseLong(dto.properties.get("padZ")));
			}
		
			graph.addNode(NodeProto.newBuilder()
				.setOpType("MaxPool")
				.addAttribute(AttributeProto.newBuilder() 
						.setName("kernel_shape")
						.setType(AttributeType.INTS)
						.addAllInts(kernel))
				.addAttribute(AttributeProto.newBuilder() 
						.setName("strides")
						.setType(AttributeType.INTS)
						.addAllInts(strides))
				.addAttribute(AttributeProto.newBuilder() 
						.setName("pads")
						.setType(AttributeType.INTS)
						.addAllInts(pads))				
				.addInput(dto.prev[0].toString())
				.addOutput(dto.id.toString())
			);
			
			break;
		}
		case "AvgPooling":
		{
			List<Long> kernel = new ArrayList<>();
			kernel.add(Long.parseLong(dto.properties.get("width")));
			if(hasProperty(dto.properties,"height")) {
				kernel.add(Long.parseLong(dto.properties.get("height")));
			}
			if(hasProperty(dto.properties,"depth")) {
				kernel.add(Long.parseLong(dto.properties.get("depth")));
			}
			
			List<Long> strides = new ArrayList<>();
			if(hasProperty(dto.properties,"strideX")) {
				strides.add(Long.parseLong(dto.properties.get("strideX")));
			}
			if(hasProperty(dto.properties,"strideY")) {
				strides.add(Long.parseLong(dto.properties.get("strideY")));
			}
			if(hasProperty(dto.properties,"strideZ")) {
				strides.add(Long.parseLong(dto.properties.get("strideZ")));
			}
			
			List<Long> pads = new ArrayList<>();
			if(hasProperty(dto.properties,"padX")) {
				pads.add(Long.parseLong(dto.properties.get("padX")));
				pads.add(Long.parseLong(dto.properties.get("padX")));
			}
			if(hasProperty(dto.properties,"padY")) {
				pads.add(Long.parseLong(dto.properties.get("padY")));
				pads.add(Long.parseLong(dto.properties.get("padY")));
			}
			if(hasProperty(dto.properties,"padZ")) {
				pads.add(Long.parseLong(dto.properties.get("padZ")));
				pads.add(Long.parseLong(dto.properties.get("padZ")));
			}
			
			graph.addNode(NodeProto.newBuilder()
				.setOpType("AveragePool")
				.addAttribute(AttributeProto.newBuilder() 
						.setName("kernel_shape")
						.setType(AttributeType.INTS)
						.addAllInts(kernel))
				.addAttribute(AttributeProto.newBuilder() 
						.setName("strides")
						.setType(AttributeType.INTS)
						.addAllInts(strides))
				.addAttribute(AttributeProto.newBuilder() 
						.setName("pads")
						.setType(AttributeType.INTS)
						.addAllInts(pads))				
				.addInput(dto.prev[0].toString())
				.addOutput(dto.id.toString())
			);
			
			break;
		}
		case "Reshape":
		{
			List<Long> dims = new ArrayList<>();
			
			int i = 0;
			do {
				long dim = Long.parseLong(dto.properties.get("dim" + i));
				dims.add(dim);
			} while(hasProperty(dto.properties, "dim" + ++i));
			
			graph.addNode(NodeProto.newBuilder()
				.setOpType("Reshape")
				.addAttribute(AttributeProto.newBuilder() 
						.setName("shape")
						.setType(AttributeType.INTS)
						.addAllInts(dims))	
				.addInput(dto.prev[0].toString())
				.addOutput(dto.id.toString())
			);
			break;
		}
		case "Zeropad":
		{
			List<Long> pads = new ArrayList<>();
			
			int i = 0;
			do {
				long pad = Long.parseLong(dto.properties.get("dim" + i));
				pads.add(pad);
				pads.add(pad); // we only do symmetric padding
			} while(hasProperty(dto.properties, "dim" + ++i));
			
			graph.addNode(NodeProto.newBuilder()
				.setOpType("Pad")
				.addAttribute(AttributeProto.newBuilder() 
						.setName("pads")
						.setType(AttributeType.INTS)
						.addAllInts(pads))	
				.addInput(dto.prev[0].toString())
				.addOutput(dto.id.toString())
			);
			break;
		}
		case "LRN":
		{
			int size = Integer.parseInt(dto.properties.get("size"));
			float alpha = Float.parseFloat(dto.properties.get("alpha"));
			float beta = Float.parseFloat(dto.properties.get("beta"));
			float k = Float.parseFloat(dto.properties.get("k"));
	
			graph.addNode(NodeProto.newBuilder()
				.setOpType("LRN")
				.addAttribute(AttributeProto.newBuilder() 
						.setName("alpha")
						.setType(AttributeType.FLOAT)
						.setF(alpha))
				.addAttribute(AttributeProto.newBuilder() 
						.setName("beta")
						.setType(AttributeType.FLOAT)
						.setF(beta))
				.addAttribute(AttributeProto.newBuilder() 
						.setName("bias")
						.setType(AttributeType.FLOAT)
						.setF(k))
				.addAttribute(AttributeProto.newBuilder() 
						.setName("size")
						.setType(AttributeType.INT)
						.setI(size))	
				.addInput(dto.prev[0].toString())
				.addOutput(dto.id.toString())
			);
			break;
		}
		default:
			throw new RuntimeException("Cannot export "+dto.type);
		}
		
		// process next modules in graph
		processed.add(dto.id);
		if(dto.next != null) {
			for(UUID nxt : dto.next) {
				convertModule(nn.modules.get(nxt));
			}
		}
	}
	
	private void convertTensor(String name, Tensor t) {
		int[] d = t.dims();
		List<Long> dims = new ArrayList<>();
		for(int i=0;i<d.length;i++) {
			dims.add((long)d[i]);
		}
		
		ByteArrayOutputStream baos = null;
		try {
			baos = new ByteArrayOutputStream(t.size()*4);
			float[] data = t.get();
			for(int i=0;i<data.length;i++) {
				writeFloat(data[i], baos);
			}
		} catch(IOException e) {
			throw new RuntimeException("Failed to write tensor data", e);
		}
		
		this.graph.addInitializer(TensorProto.newBuilder()
				.setName(name)
				.setDataType(DataType.FLOAT)
				.addAllDims(dims)
				.setRawData(ByteString.copyFrom(baos.toByteArray())));
	}
	
	private void writeFloat(float v, ByteArrayOutputStream out) throws IOException {
	    int f = Float.floatToIntBits(v);
	    out.write(0xFF & f);
	    out.write(0xFF & (f >> 8));
	    out.write(0xFF & (f >> 16));
	    out.write(0xFF & (f >> 24));
	}
	
	private boolean hasProperty(Map<String, String> config, String property){
		String value = (String) config.get(property);
		if(value==null){
			return false;
		} else if(value.isEmpty()){
			return false;
		}
		return true;
	}
}