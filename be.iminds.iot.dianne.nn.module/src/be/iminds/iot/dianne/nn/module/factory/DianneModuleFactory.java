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
package be.iminds.iot.dianne.nn.module.factory;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;

import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Component;

import be.iminds.iot.dianne.api.nn.module.AbstractModule;
import be.iminds.iot.dianne.api.nn.module.Module;
import be.iminds.iot.dianne.api.nn.module.dto.ModuleDTO;
import be.iminds.iot.dianne.api.nn.module.dto.ModulePropertyDTO;
import be.iminds.iot.dianne.api.nn.module.dto.ModuleTypeDTO;
import be.iminds.iot.dianne.api.nn.module.factory.ModuleFactory;
import be.iminds.iot.dianne.api.nn.module.factory.ModuleTypeNotSupportedException;
import be.iminds.iot.dianne.nn.module.activation.LogSoftmax;
import be.iminds.iot.dianne.nn.module.activation.PReLU;
import be.iminds.iot.dianne.nn.module.activation.ReLU;
import be.iminds.iot.dianne.nn.module.activation.Sigmoid;
import be.iminds.iot.dianne.nn.module.activation.SoftPlus;
import be.iminds.iot.dianne.nn.module.activation.Softmax;
import be.iminds.iot.dianne.nn.module.activation.Tanh;
import be.iminds.iot.dianne.nn.module.activation.Threshold;
import be.iminds.iot.dianne.nn.module.fork.Duplicate;
import be.iminds.iot.dianne.nn.module.fork.Grid;
import be.iminds.iot.dianne.nn.module.fork.Split;
import be.iminds.iot.dianne.nn.module.io.InputImpl;
import be.iminds.iot.dianne.nn.module.io.OutputImpl;
import be.iminds.iot.dianne.nn.module.join.Accumulate;
import be.iminds.iot.dianne.nn.module.join.Average;
import be.iminds.iot.dianne.nn.module.join.Concat;
import be.iminds.iot.dianne.nn.module.join.Multiply;
import be.iminds.iot.dianne.nn.module.layer.AvgPooling;
import be.iminds.iot.dianne.nn.module.layer.Convolution;
import be.iminds.iot.dianne.nn.module.layer.FullConvolution;
import be.iminds.iot.dianne.nn.module.layer.Linear;
import be.iminds.iot.dianne.nn.module.layer.MaskedMaxPooling;
import be.iminds.iot.dianne.nn.module.layer.MaxPooling;
import be.iminds.iot.dianne.nn.module.layer.MaxUnpooling;
import be.iminds.iot.dianne.nn.module.layer.Narrow;
import be.iminds.iot.dianne.nn.module.layer.Reshape;
import be.iminds.iot.dianne.nn.module.layer.Zeropad;
import be.iminds.iot.dianne.nn.module.preprocessing.Denormalization;
import be.iminds.iot.dianne.nn.module.preprocessing.Frame;
import be.iminds.iot.dianne.nn.module.preprocessing.Normalization;
import be.iminds.iot.dianne.nn.module.preprocessing.Scale;
import be.iminds.iot.dianne.nn.module.regularization.BatchNormalization;
import be.iminds.iot.dianne.nn.module.regularization.DropPath;
import be.iminds.iot.dianne.nn.module.regularization.Dropout;
import be.iminds.iot.dianne.nn.module.vae.GaussianSampler;
import be.iminds.iot.dianne.nn.module.vae.MultivariateGaussian;
import be.iminds.iot.dianne.tensor.Tensor;

@Component(property={"aiolos.proxy=false"})
public class DianneModuleFactory implements ModuleFactory {

	private final Map<String, ModuleTypeDTO> supportedModules = new HashMap<String, ModuleTypeDTO>();
	
	@Activate
	void activate(){
		// build list of supported modules
		// TODO use reflection for this?
		addSupportedType( new ModuleTypeDTO("Multivariate Gaussian", "Variational", true, 
				new ModulePropertyDTO("Size", "size", Integer.class.getName()),
				new ModulePropertyDTO("Mean Activation", "meanActivation", String.class.getName()),
				new ModulePropertyDTO("Stdev Activation", "stdevActivation", String.class.getName())));
		
		addSupportedType( new ModuleTypeDTO("Gaussian Sampler", "Variational", true, 
				new ModulePropertyDTO("Size", "size", Integer.class.getName())));
		
		
		addSupportedType( new ModuleTypeDTO("BatchNormalization", "Regularization", true, 
				new ModulePropertyDTO("Size", "size", Integer.class.getName())));
		
		addSupportedType( new ModuleTypeDTO("Dropout", "Regularization", true, 
				new ModulePropertyDTO("Dropout rate", "rate", Float.class.getName())));

		addSupportedType( new ModuleTypeDTO("DropPath", "Regularization", true, 
				new ModulePropertyDTO("DropPath rate", "rate", Float.class.getName())));
		
		addSupportedType( new ModuleTypeDTO("Linear", "Layer", true, 
					new ModulePropertyDTO("Input size", "input", Integer.class.getName()),
					new ModulePropertyDTO("Output size", "output", Integer.class.getName())));

		addSupportedType(new ModuleTypeDTO("Sigmoid", "Activation", false));
		
		addSupportedType(new ModuleTypeDTO("Tanh", "Activation", false));
		
		addSupportedType(new ModuleTypeDTO("SoftPlus", "Activation", false,
					new ModulePropertyDTO("Beta value", "beta", Float.class.getName())));
		
		addSupportedType(new ModuleTypeDTO("ELU", "Activation", false,
				new ModulePropertyDTO("Alpha value", "alpha", Float.class.getName())));
		
		addSupportedType(new ModuleTypeDTO("Softmax", "Activation", false));
		
		addSupportedType(new ModuleTypeDTO("LogSoftmax", "Activation", false));	
	
		addSupportedType(new ModuleTypeDTO("ReLU", "Activation", false));

		addSupportedType(new ModuleTypeDTO("PReLU", "Activation", true));
		
		addSupportedType(new ModuleTypeDTO("Threshold", "Activation", false,
					new ModulePropertyDTO("Threshold value", "thresh", Float.class.getName()),
					new ModulePropertyDTO("Replacement value", "val", Float.class.getName())));
		
		addSupportedType(new ModuleTypeDTO("Input", "Input-Output", false));
		
		addSupportedType(new ModuleTypeDTO("Output", "Input-Output", false));

		addSupportedType(new ModuleTypeDTO("Duplicate", "Fork", false,
				new ModulePropertyDTO("Wait for all", "waitForAll", Boolean.class.getName())));
		
		addSupportedType(new ModuleTypeDTO("Accumulate", "Join", false,
				new ModulePropertyDTO("Wait for all", "waitForAll", Boolean.class.getName())));
		
		addSupportedType(new ModuleTypeDTO("Average", "Join", false,
				new ModulePropertyDTO("Wait for all", "waitForAll", Boolean.class.getName())));
		
		addSupportedType(new ModuleTypeDTO("Multiply", "Join", false,
				new ModulePropertyDTO("Wait for all", "waitForAll", Boolean.class.getName())));
		
		addSupportedType(new ModuleTypeDTO("Split", "Fork", false,
				new ModulePropertyDTO("Wait for all", "waitForAll", Boolean.class.getName()),
				new ModulePropertyDTO("Split dimension", "dim", Integer.class.getName())));
		
		addSupportedType(new ModuleTypeDTO("Concat", "Join", false,
				new ModulePropertyDTO("Wait for all", "waitForAll", Boolean.class.getName()),
				new ModulePropertyDTO("Concat dimension", "dim", Integer.class.getName())));
			
		addSupportedType(new ModuleTypeDTO("Grid", "Fork", false,
				new ModulePropertyDTO("X", "x", Integer.class.getName()),
				new ModulePropertyDTO("Y", "y", Integer.class.getName()),
				new ModulePropertyDTO("Stride X", "strideX", Integer.class.getName()),
				new ModulePropertyDTO("Stride Y", "strideY", Integer.class.getName())));
			
		addSupportedType(new ModuleTypeDTO("Convolution", "Layer", true, 
				new ModulePropertyDTO("Input planes", "noInputPlanes", Integer.class.getName()),
				new ModulePropertyDTO("Output planes", "noOutputPlanes", Integer.class.getName()),
				new ModulePropertyDTO("Kernel width", "kernelWidth", Integer.class.getName()),
				new ModulePropertyDTO("Kernel height", "kernelHeight", Integer.class.getName()),
				new ModulePropertyDTO("Kernel depth", "kernelDepth", Integer.class.getName()),
				new ModulePropertyDTO("Stride X", "strideX", Integer.class.getName()),
				new ModulePropertyDTO("Stride Y", "strideY", Integer.class.getName()),
				new ModulePropertyDTO("Stride Z", "strideZ", Integer.class.getName()),
				new ModulePropertyDTO("Pad X", "padX", Integer.class.getName()),
				new ModulePropertyDTO("Pad Y", "padY", Integer.class.getName()),
				new ModulePropertyDTO("Pad Z", "padZ", Integer.class.getName())));
			
		addSupportedType(new ModuleTypeDTO("FullConvolution", "Layer", true, 
				new ModulePropertyDTO("Input planes", "noInputPlanes", Integer.class.getName()),
				new ModulePropertyDTO("Output planes", "noOutputPlanes", Integer.class.getName()),
				new ModulePropertyDTO("Kernel width", "kernelWidth", Integer.class.getName()),
				new ModulePropertyDTO("Kernel height", "kernelHeight", Integer.class.getName()),
				new ModulePropertyDTO("Kernel depth", "kernelDepth", Integer.class.getName()),
				new ModulePropertyDTO("Stride X", "strideX", Integer.class.getName()),
				new ModulePropertyDTO("Stride Y", "strideY", Integer.class.getName()),
				new ModulePropertyDTO("Stride Z", "strideZ", Integer.class.getName()),
				new ModulePropertyDTO("Pad X", "padX", Integer.class.getName()),
				new ModulePropertyDTO("Pad Y", "padY", Integer.class.getName()),
				new ModulePropertyDTO("Pad Z", "padZ", Integer.class.getName())));
		
		addSupportedType(new ModuleTypeDTO("MaxPooling" , "Layer", false, 
				new ModulePropertyDTO("Width", "width", Integer.class.getName()),
				new ModulePropertyDTO("Height", "height", Integer.class.getName()),
				new ModulePropertyDTO("Depth", "depth", Integer.class.getName()),
				new ModulePropertyDTO("Stride X", "strideX", Integer.class.getName()),
				new ModulePropertyDTO("Stride Y", "strideY", Integer.class.getName()),
				new ModulePropertyDTO("Stride Z", "strideZ", Integer.class.getName())));

		addSupportedType(new ModuleTypeDTO("MaxUnpooling" , "Layer", false, 
				new ModulePropertyDTO("Width", "width", Integer.class.getName()),
				new ModulePropertyDTO("Height", "height", Integer.class.getName()),
				new ModulePropertyDTO("Depth", "depth", Integer.class.getName()),
				new ModulePropertyDTO("Stride X", "strideX", Integer.class.getName()),
				new ModulePropertyDTO("Stride Y", "strideY", Integer.class.getName()),
				new ModulePropertyDTO("Stride Z", "strideZ", Integer.class.getName())));
		
		addSupportedType(new ModuleTypeDTO("AvgPooling" , "Layer", false, 
				new ModulePropertyDTO("Width", "width", Integer.class.getName()),
				new ModulePropertyDTO("Height", "height", Integer.class.getName()),
				new ModulePropertyDTO("Depth", "depth", Integer.class.getName()),
				new ModulePropertyDTO("Stride X", "strideX", Integer.class.getName()),
				new ModulePropertyDTO("Stride Y", "strideY", Integer.class.getName()),
				new ModulePropertyDTO("Stride Z", "strideZ", Integer.class.getName())));
		
		addSupportedType(new ModuleTypeDTO("Normalization", "Preprocessing", true));
		addSupportedType(new ModuleTypeDTO("Denormalization", "Preprocessing", true));
		
		addSupportedType(new ModuleTypeDTO("Narrow", "Layer", false, 
				new ModulePropertyDTO("Index dim 0", "index0", Integer.class.getName()),
				new ModulePropertyDTO("Size dim 0", "size0", Integer.class.getName()),
				new ModulePropertyDTO("Index dim 1", "index1", Integer.class.getName()),
				new ModulePropertyDTO("Size dim 1", "size1", Integer.class.getName()),
				new ModulePropertyDTO("Index dim 2", "index2", Integer.class.getName()),
				new ModulePropertyDTO("Size dim 2", "size2", Integer.class.getName())));
		
		addSupportedType(new ModuleTypeDTO("Scale", "Preprocessing", false, 
				new ModulePropertyDTO("Dim 0", "dim0", Integer.class.getName()),
				new ModulePropertyDTO("Dim 1", "dim1", Integer.class.getName()),
				new ModulePropertyDTO("Dim 2", "dim2", Integer.class.getName()),
				new ModulePropertyDTO("Factor 0", "factor0", Float.class.getName()),
				new ModulePropertyDTO("Factor 1", "factor1", Float.class.getName()),
				new ModulePropertyDTO("Factor 2", "factor2", Float.class.getName())));
		
		addSupportedType(new ModuleTypeDTO("Frame", "Preprocessing", false, 
				new ModulePropertyDTO("Dim 0", "dim0", Integer.class.getName()),
				new ModulePropertyDTO("Dim 1", "dim1", Integer.class.getName()),
				new ModulePropertyDTO("Dim 2", "dim2", Integer.class.getName())));	
		
		addSupportedType(new ModuleTypeDTO("Reshape", "Layer", false, 
				new ModulePropertyDTO("Dim 0", "dim0", Integer.class.getName()),
				new ModulePropertyDTO("Dim 1", "dim1", Integer.class.getName()),
				new ModulePropertyDTO("Dim 2", "dim2", Integer.class.getName())));

		addSupportedType(new ModuleTypeDTO("Zeropad", "Layer", false, 
				new ModulePropertyDTO("Dim 0", "dim0", Integer.class.getName()),
				new ModulePropertyDTO("Dim 1", "dim1", Integer.class.getName()),
				new ModulePropertyDTO("Dim 2", "dim2", Integer.class.getName())));
		
		addSupportedType(new ModuleTypeDTO("Masked Maxpooling", "Layer", true, 
				new ModulePropertyDTO("Inputs", "noInputs", Integer.class.getName()),
				new ModulePropertyDTO("Masks", "masks", String.class.getName())));

	}
	
	
	@Override
	public Module createModule(ModuleDTO dto)
			throws InstantiationException {
		return createModule(dto, null);
	}
	
	@Override
	public Module createModule(ModuleDTO dto, Tensor parameters)
			throws InstantiationException {

		AbstractModule module = null;
		
		// TODO use reflection for this?
		// for now just hard code an if/else for each known module
		String type = dto.type;
		UUID id = dto.id;

		switch(type){
		case "Linear":
		{
			int input = Integer.parseInt(dto.properties.get("input"));
			int output = Integer.parseInt(dto.properties.get("output"));
			
			if(parameters!=null){
				module = new Linear(id, parameters, input, output);
			} else {
				module = new Linear(id, input, output);
			}
			break;
		}
		case "BatchNormalization":
		{
			int size = Integer.parseInt(dto.properties.get("size"));
			
			if(parameters!=null){
				module = new BatchNormalization(id, parameters, size);
			} else {
				module = new BatchNormalization(id, size);
			}
			break;
		}
		case "Dropout":
		{
			float rate = Float.parseFloat(dto.properties.get("rate"));
			module = new Dropout(id, rate);
			break;
		}
		case "DropPath":
		{
			float rate = Float.parseFloat(dto.properties.get("rate"));
			module = new DropPath(id, rate);
			break;
		}
		case "Tanh":
		{
			module = new Tanh(id);
			break;
		}
		case "Sigmoid":
		{
			module = new Sigmoid(id);
			break;
		}
		case "SoftPlus":
		{
			module = hasProperty(dto.properties, "beta") ? 
					new SoftPlus(id, Float.parseFloat(dto.properties.get("beta"))) : new SoftPlus(id);
			break;
		}
		case "ELU":
		{
			module = hasProperty(dto.properties, "alpha") ? 
					new SoftPlus(id, Float.parseFloat(dto.properties.get("alpha"))) : new SoftPlus(id);
			break;
		}
		case "Softmax":
		{
			module = new Softmax(id);
			break;
		}
		case "LogSoftmax":
		{
			module = new LogSoftmax(id);
			break;
		}
		case "ReLU":
		{
			module = new ReLU(id);
			break;
		}
		case "PReLU":
		{
			module = new PReLU(id);
			if(parameters!=null){
				((PReLU)module).setParameters(parameters);
			}
			break;
		}
		case "Threshold":
		{
			float thresh = Float.parseFloat(dto.properties.get("thresh"));
			float val = Float.parseFloat(dto.properties.get("val"));
			
			module = new Threshold(id, thresh, val);
			break;
		}
		case "Input":
		{
			module = new InputImpl(id);
			break;
		}
		case "Output":
		{
			module = new OutputImpl(id);
			break;
		}
		case "Duplicate":
		{
			module = hasProperty(dto.properties, "waitForAll") ? 
					new Duplicate(id, Boolean.parseBoolean(dto.properties.get("waitForAll"))) : new Duplicate(id);
			break;
		}
		case "Accumulate":
		{
			module = hasProperty(dto.properties, "waitForAll") ? 
					new Accumulate(id, Boolean.parseBoolean(dto.properties.get("waitForAll"))) : new Accumulate(id);
			break;
		}
		case "Average":
		{
			module = hasProperty(dto.properties, "waitForAll") ? 
					new Average(id, Boolean.parseBoolean(dto.properties.get("waitForAll"))) : new Average(id);
			break;
		}		
		case "Multiply":
		{
			module = hasProperty(dto.properties, "waitForAll") ? 
					new Multiply(id, Boolean.parseBoolean(dto.properties.get("waitForAll"))) : new Multiply(id);
			break;
		}
		case "Split":
		{
			int dim = Integer.parseInt(dto.properties.get("dim"));
			module = hasProperty(dto.properties, "waitForAll") ? 
					new Split(id, Boolean.parseBoolean(dto.properties.get("waitForAll")), dim) : new Split(id, dim);
			break;
		}
		case "Concat":
		{
			int dim = Integer.parseInt(dto.properties.get("dim"));
			module = hasProperty(dto.properties, "waitForAll") ? 
					new Concat(id, Boolean.parseBoolean(dto.properties.get("waitForAll")), dim) : new Concat(id, dim);
			break;
		}
		case "Grid":
		{
			int x = Integer.parseInt(dto.properties.get("x"));
			int y = Integer.parseInt(dto.properties.get("y"));
			
			int strideX = hasProperty(dto.properties,"strideX") ? Integer.parseInt(dto.properties.get("strideX")) : 1;
			int strideY = hasProperty(dto.properties,"strideY") ? Integer.parseInt(dto.properties.get("strideY")) : 1;

			module = new Grid(id, x, y, strideX, strideY);
			break;
		}
		case "Convolution":
		{
			int noInputPlanes = Integer.parseInt(dto.properties.get("noInputPlanes"));
			int noOutputPlanes = Integer.parseInt(dto.properties.get("noOutputPlanes"));
			int kernelWidth = Integer.parseInt(dto.properties.get("kernelWidth"));
			int kernelHeight = hasProperty(dto.properties,"kernelHeight") ? Integer.parseInt(dto.properties.get("kernelHeight")) : 1;
			int kernelDepth = hasProperty(dto.properties,"kernelDepth") ? Integer.parseInt(dto.properties.get("kernelDepth")) : 1;
			
			int strideX = 1, strideY = 1, strideZ = 1;
			int padX = 0, padY = 0, padZ = 0;
			
			// remain backward compatible with the old boolean for full conv zero padding
			boolean pad = hasProperty(dto.properties,"pad") ? Boolean.parseBoolean(dto.properties.get("pad")) : false;
			if(pad){
				padX = (kernelWidth-1)/2;
				padY = (kernelHeight-1)/2;
				padZ = (kernelDepth-1)/2;
			}
			
			strideX = hasProperty(dto.properties,"strideX") ? Integer.parseInt(dto.properties.get("strideX")) : strideX;
			strideY = hasProperty(dto.properties,"strideY") ? Integer.parseInt(dto.properties.get("strideY")) : strideY;
			strideZ = hasProperty(dto.properties,"strideZ") ? Integer.parseInt(dto.properties.get("strideZ")) : strideZ;

			padX = hasProperty(dto.properties,"padX") ? Integer.parseInt(dto.properties.get("padX")) : padX;
			padY = hasProperty(dto.properties,"padY") ? Integer.parseInt(dto.properties.get("padY")) : padY;
			padZ = hasProperty(dto.properties,"padZ") ? Integer.parseInt(dto.properties.get("padZ")) : padZ;

			
			if(hasProperty(dto.properties,"kernelDepth")){
				// volumetric
				if(parameters!=null){
					module = new Convolution(id, parameters, noInputPlanes, noOutputPlanes, kernelWidth, kernelHeight, kernelDepth, strideX, strideY, strideZ, padX, padY, padZ);
				} else {
					module = new Convolution(id, noInputPlanes, noOutputPlanes, kernelWidth, kernelHeight, kernelDepth, strideX, strideY, strideZ, padX, padY, padZ);
				}

			} else if(hasProperty(dto.properties,"kernelHeight")){
				// spatial
				if(parameters!=null){
					module = new Convolution(id, parameters, noInputPlanes, noOutputPlanes, kernelWidth, kernelHeight, strideX, strideY, padX, padY);
				} else {
					module = new Convolution(id, noInputPlanes, noOutputPlanes, kernelWidth, kernelHeight, strideX, strideY, padX, padY);
				}
			} else {
				// temporal
				if(parameters!=null){
					module = new Convolution(id, parameters, noInputPlanes, noOutputPlanes, kernelWidth, strideX);
				} else {
					module = new Convolution(id, noInputPlanes, noOutputPlanes, kernelWidth, strideX);
				}
			}

			break;
		}
		case "FullConvolution":
		{
			int noInputPlanes = Integer.parseInt(dto.properties.get("noInputPlanes"));
			int noOutputPlanes = Integer.parseInt(dto.properties.get("noOutputPlanes"));
			int kernelWidth = Integer.parseInt(dto.properties.get("kernelWidth"));
			int kernelHeight = hasProperty(dto.properties,"kernelHeight") ? Integer.parseInt(dto.properties.get("kernelHeight")) : 1;
			int kernelDepth = hasProperty(dto.properties,"kernelDepth") ? Integer.parseInt(dto.properties.get("kernelDepth")) : 1;
			
			int strideX = 1, strideY = 1, strideZ = 1;
			int padX = 0, padY = 0, padZ = 0;
			
			// remain backward compatible with the old boolean for full conv zero padding
			boolean pad = hasProperty(dto.properties,"pad") ? Boolean.parseBoolean(dto.properties.get("pad")) : false;
			if(pad){
				padX = (kernelWidth-1)/2;
				padY = (kernelHeight-1)/2;
				padZ = (kernelDepth-1)/2;
			}
			
			strideX = hasProperty(dto.properties,"strideX") ? Integer.parseInt(dto.properties.get("strideX")) : strideX;
			strideY = hasProperty(dto.properties,"strideY") ? Integer.parseInt(dto.properties.get("strideY")) : strideY;
			strideZ = hasProperty(dto.properties,"strideZ") ? Integer.parseInt(dto.properties.get("strideZ")) : strideZ;

			padX = hasProperty(dto.properties,"padX") ? Integer.parseInt(dto.properties.get("padX")) : padX;
			padY = hasProperty(dto.properties,"padY") ? Integer.parseInt(dto.properties.get("padY")) : padY;
			padZ = hasProperty(dto.properties,"padZ") ? Integer.parseInt(dto.properties.get("padZ")) : padZ;

			
			if(hasProperty(dto.properties,"kernelDepth")){
				// volumetric
				if(parameters!=null){
					module = new FullConvolution(id, parameters, noInputPlanes, noOutputPlanes, kernelWidth, kernelHeight, kernelDepth, strideX, strideY, strideZ, padX, padY, padZ);
				} else {
					module = new FullConvolution(id, noInputPlanes, noOutputPlanes, kernelWidth, kernelHeight, kernelDepth, strideX, strideY, strideZ, padX, padY, padZ);
				}

			} else if(hasProperty(dto.properties,"kernelHeight")){
				// spatial
				if(parameters!=null){
					module = new FullConvolution(id, parameters, noInputPlanes, noOutputPlanes, kernelWidth, kernelHeight, strideX, strideY, padX, padY);
				} else {
					module = new FullConvolution(id, noInputPlanes, noOutputPlanes, kernelWidth, kernelHeight, strideX, strideY, padX, padY);
				}
			} else {
				throw new RuntimeException("Temporal Full convultion not implemented");
			}

			break;
		}
		case "MaxPooling":
		{
			int width = Integer.parseInt(dto.properties.get("width"));
			int height = hasProperty(dto.properties, "height") ? Integer.parseInt(dto.properties.get("height")) : 1;
			int depth =  hasProperty(dto.properties, "depth") ? Integer.parseInt(dto.properties.get("depth")) : 1;

			int sx = hasProperty(dto.properties, "strideX") ? Integer.parseInt(dto.properties.get("strideX")) : width;
			int sy = hasProperty(dto.properties,"strideY") ? Integer.parseInt(dto.properties.get("strideY")) : height;
			int sz = hasProperty(dto.properties,"strideZ") ? Integer.parseInt(dto.properties.get("strideZ")) : depth;

			if(hasProperty(dto.properties, "depth")){
				module = new MaxPooling(id, width, height, depth, sx, sy, sz);
			} else if(hasProperty(dto.properties, "height")){
				module = new MaxPooling(id, width, height, sx, sy);
			} else {
				module = new MaxPooling(id, width, sx);
			}
			
			break;
		}
		case "MaxUnpooling":
		{
			int width = Integer.parseInt(dto.properties.get("width"));
			int height = hasProperty(dto.properties, "height") ? Integer.parseInt(dto.properties.get("height")) : 1;
			int depth =  hasProperty(dto.properties, "depth") ? Integer.parseInt(dto.properties.get("depth")) : 1;

			int sx = hasProperty(dto.properties, "strideX") ? Integer.parseInt(dto.properties.get("strideX")) : width;
			int sy = hasProperty(dto.properties,"strideY") ? Integer.parseInt(dto.properties.get("strideY")) : height;
			int sz = hasProperty(dto.properties,"strideZ") ? Integer.parseInt(dto.properties.get("strideZ")) : depth;

			if(hasProperty(dto.properties, "depth")){
				module = new MaxUnpooling(id, width, height, depth, sx, sy, sz);
			} else if(hasProperty(dto.properties, "height")){
				module = new MaxUnpooling(id, width, height, sx, sy);
			} else {
				module = new MaxUnpooling(id, width, sx);
			}
			
			break;
		}		
		case "AvgPooling":
		{
			int width = Integer.parseInt(dto.properties.get("width"));
			int height = hasProperty(dto.properties, "height") ? Integer.parseInt(dto.properties.get("height")) : 1;
			int depth =  hasProperty(dto.properties, "depth") ? Integer.parseInt(dto.properties.get("depth")) : 1;

			int sx = hasProperty(dto.properties, "strideX") ? Integer.parseInt(dto.properties.get("strideX")) : width;
			int sy = hasProperty(dto.properties,"strideY") ? Integer.parseInt(dto.properties.get("strideY")) : height;
			int sz = hasProperty(dto.properties,"strideZ") ? Integer.parseInt(dto.properties.get("strideZ")) : depth;

			if(hasProperty(dto.properties, "depth")){
				module = new AvgPooling(id, width, height, depth, sx, sy, sz);
			} else if(hasProperty(dto.properties, "height")){
				module = new AvgPooling(id, width, height, sx, sy);
			} else {
				module = new AvgPooling(id, width, sx);
			}
			break;
		}
		case "Normalization":
		{
			module = new Normalization(id);
			if(parameters!=null){
				((Normalization)module).setParameters(parameters);
			}
			break;
		}
		case "Denormalization":
		{
			module = new Denormalization(id);
			if(parameters!=null){
				((Denormalization)module).setParameters(parameters);
			}
			break;
		}
		case "Narrow":
		{
			int[] ranges;
			
			int index0 = Integer.parseInt(dto.properties.get("index0"));
			int size0 = Integer.parseInt(dto.properties.get("size0"));
			
			if(hasProperty(dto.properties, "index1")){
				int index1 = Integer.parseInt(dto.properties.get("index1"));
				int size1 = Integer.parseInt(dto.properties.get("size1"));
			
				if(hasProperty(dto.properties, "index2")){
					int index2 = Integer.parseInt(dto.properties.get("index2"));
					int size2 = Integer.parseInt(dto.properties.get("size2"));
					
					ranges = new int[]{index0, size0, index1, size1, index2, size2};
				} else {
					ranges = new int[]{index0, size0, index1, size1};
				}
			} else {
				ranges = new int[]{index0, size0};
			}
			
			module = new Narrow(id, ranges);
			break;
		}
		case "Scale":
		{
			if(hasProperty(dto.properties, "dim0")){
				List<Integer> dims = new ArrayList<>();
				
				int i = 0;
				do {
					int dim = Integer.parseInt(dto.properties.get("dim" + i));
					dims.add(dim);
				} while(hasProperty(dto.properties, "dim" + ++i));
				
				module = new Scale(id, dims.stream().mapToInt(d -> d).toArray());
				break;
				
			} else if(hasProperty(dto.properties, "factor0")){
				List<Float> factors = new ArrayList<>();
				
				int i = 0;
				do {
					float factor = Float.parseFloat(dto.properties.get("factor" + i));
					factors.add(factor);
				} while(hasProperty(dto.properties, "factor" + ++i));
				
				module = new Scale(id, factors.stream().mapToDouble(f -> f).toArray());
				break;
			}	 			
		}
		case "Frame":
		{
			List<Integer> dims = new ArrayList<>();
			
			int i = 0;
			do {
				int dim = Integer.parseInt(dto.properties.get("dim" + i));
				dims.add(dim);
			} while(hasProperty(dto.properties, "dim" + ++i));
			
			module = new Frame(id, dims.stream().mapToInt(d -> d).toArray());
			break;
		}
		case "Reshape":
		{
			List<Integer> dims = new ArrayList<>();
			
			int i = 0;
			do {
				int dim = Integer.parseInt(dto.properties.get("dim" + i));
				dims.add(dim);
			} while(hasProperty(dto.properties, "dim" + ++i));
			
			module = new Reshape(id, dims.stream().mapToInt(d -> d).toArray());
			break;
		}
		case "Zeropad":
		{
			List<Integer> dims = new ArrayList<>();
			
			int i = 0;
			do {
				int dim = Integer.parseInt(dto.properties.get("dim" + i));
				dims.add(dim);
			} while(hasProperty(dto.properties, "dim" + ++i));
			
			module = new Zeropad(id, dims.stream().mapToInt(d -> d).toArray());
			break;
		}
		case "Masked MaxPooling":
		{
			int noInputs = Integer.parseInt(dto.properties.get("noInputs"));
			String masks = dto.properties.get("masks");
			
			module = new MaskedMaxPooling(id, noInputs, masks);
			break;
		}
		case "Multivariate Gaussian":
		{
			int size = Integer.parseInt(dto.properties.get("size"));
			String meanActivation = dto.properties.get("meanActivation");
			String stdevActivation = dto.properties.get("stdevActivation");
			
			module = new MultivariateGaussian(id, size, meanActivation, stdevActivation);
			break;
		}
		case "Gaussian Sampler":
		{
			int size = Integer.parseInt(dto.properties.get("size"));
			module = new GaussianSampler(id, size);
			break;
		}
		default:
			throw new InstantiationException("Could not instantiate module of type "+type);
		}
		
		return module;
	}

	@Override
	public List<ModuleTypeDTO> getAvailableModuleTypes() {
		return new ArrayList<ModuleTypeDTO>(supportedModules.values());
	}

	@Override
	public ModuleTypeDTO getModuleType(String name) {
		return supportedModules.get(name);
	}
	
	@Override
	public int parameterSize(ModuleDTO m) throws ModuleTypeNotSupportedException {
		if(!supportedModules.containsKey(m.type))
			throw new ModuleTypeNotSupportedException(m.type);
		
		int size = 0;
		switch(m.type){
		case "Linear":
			int inSize = Integer.parseInt(m.properties.get("input"));
			int outSize = Integer.parseInt(m.properties.get("output"));
			size = outSize*(inSize+1);
			break;
		case "Convolution":
		case "FullConvolution":
			int noInputPlanes = Integer.parseInt(m.properties.get("noInputPlanes"));
			int noOutputPlanes = Integer.parseInt(m.properties.get("noOutputPlanes"));
			int kernelWidth = Integer.parseInt(m.properties.get("kernelWidth"));
			int kernelHeight = hasProperty(m.properties, "kernelHeight") ? Integer.parseInt(m.properties.get("kernelHeight")) : 1;
			int kernelDepth = hasProperty(m.properties, "kernelDepth") ? Integer.parseInt(m.properties.get("kernelDepth")) : 1;

			size = noOutputPlanes*noInputPlanes*kernelWidth*kernelHeight*kernelDepth+noOutputPlanes;
			break;
		case "PReLU":
			size = 1;
			break;
		case "BatchNormalization":
			size = Integer.parseInt(m.properties.get("size"))*4;
			break;	
		}
		return size;
	}


	@Override
	public int memorySize(ModuleDTO m) throws ModuleTypeNotSupportedException {
		if(!supportedModules.containsKey(m.type))
			throw new ModuleTypeNotSupportedException(m.type);
		
		return 0;
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
	
	private void addSupportedType(ModuleTypeDTO t){
		supportedModules.put(t.type, t);
	}
	
}
