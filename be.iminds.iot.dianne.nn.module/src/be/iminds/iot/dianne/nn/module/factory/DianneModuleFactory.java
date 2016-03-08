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
import org.osgi.service.component.annotations.Reference;

import be.iminds.iot.dianne.api.nn.module.AbstractModule;
import be.iminds.iot.dianne.api.nn.module.Module;
import be.iminds.iot.dianne.api.nn.module.dto.ModuleDTO;
import be.iminds.iot.dianne.api.nn.module.dto.ModulePropertyDTO;
import be.iminds.iot.dianne.api.nn.module.dto.ModuleTypeDTO;
import be.iminds.iot.dianne.api.nn.module.factory.ModuleFactory;
import be.iminds.iot.dianne.nn.module.activation.PReLU;
import be.iminds.iot.dianne.nn.module.activation.ReLU;
import be.iminds.iot.dianne.nn.module.activation.Sigmoid;
import be.iminds.iot.dianne.nn.module.activation.Softmax;
import be.iminds.iot.dianne.nn.module.activation.Tanh;
import be.iminds.iot.dianne.nn.module.activation.Threshold;
import be.iminds.iot.dianne.nn.module.fork.Duplicate;
import be.iminds.iot.dianne.nn.module.fork.Grid;
import be.iminds.iot.dianne.nn.module.fork.Split;
import be.iminds.iot.dianne.nn.module.io.InputImpl;
import be.iminds.iot.dianne.nn.module.io.OutputImpl;
import be.iminds.iot.dianne.nn.module.join.Accumulate;
import be.iminds.iot.dianne.nn.module.join.Concat;
import be.iminds.iot.dianne.nn.module.layer.Dropout;
import be.iminds.iot.dianne.nn.module.join.Multiply;
import be.iminds.iot.dianne.nn.module.layer.Linear;
import be.iminds.iot.dianne.nn.module.layer.MaskedMaxPooling;
import be.iminds.iot.dianne.nn.module.layer.SpatialConvolution;
import be.iminds.iot.dianne.nn.module.layer.SpatialMaxPooling;
import be.iminds.iot.dianne.nn.module.preprocessing.Frame;
import be.iminds.iot.dianne.nn.module.preprocessing.Narrow;
import be.iminds.iot.dianne.nn.module.preprocessing.Normalization;
import be.iminds.iot.dianne.nn.module.preprocessing.Scale;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;

@Component(property={"aiolos.export=false"})
public class DianneModuleFactory implements ModuleFactory {

	private TensorFactory factory;
	
	private final Map<String, ModuleTypeDTO> supportedModules = new HashMap<String, ModuleTypeDTO>();
	
	@Activate
	void activate(){
		// build list of supported modules
		// TODO use reflection for this?
		
		addSupportedType( new ModuleTypeDTO("Linear", "Layer", true, 
					new ModulePropertyDTO("Input size", "input", Integer.class.getName()),
					new ModulePropertyDTO("Output size", "output", Integer.class.getName())));

		addSupportedType( new ModuleTypeDTO("Dropout", "Layer", true, 
				new ModulePropertyDTO("Dropout rate", "dropout", Float.class.getName())));
		
		addSupportedType(new ModuleTypeDTO("Sigmoid", "Activation", false));
		
		addSupportedType(new ModuleTypeDTO("Tanh", "Activation", false));
		
		addSupportedType(new ModuleTypeDTO("Softmax", "Activation", false));	
	
		addSupportedType(new ModuleTypeDTO("ReLU", "Activation", false));

		addSupportedType(new ModuleTypeDTO("PReLU", "Activation", true));
		
		addSupportedType(new ModuleTypeDTO("Threshold", "Activation", false,
					new ModulePropertyDTO("Threshold value", "thresh", Float.class.getName()),
					new ModulePropertyDTO("Replacement value", "val", Float.class.getName())));
		
		addSupportedType(new ModuleTypeDTO("Input", "Input-Output", false));
		
		addSupportedType(new ModuleTypeDTO("Output", "Input-Output", false));

		addSupportedType(new ModuleTypeDTO("Duplicate", "Fork", false));
		
		addSupportedType(new ModuleTypeDTO("Accumulate", "Join", false));
		
		addSupportedType(new ModuleTypeDTO("Multiply", "Join", false));
		
		addSupportedType(new ModuleTypeDTO("Split", "Fork", false));
		
		addSupportedType(new ModuleTypeDTO("Concat", "Join", false));
			
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
				new ModulePropertyDTO("Stride X", "strideX", Integer.class.getName()),
				new ModulePropertyDTO("Stride Y", "strideY", Integer.class.getName()),
				new ModulePropertyDTO("Add zero padding", "pad", Boolean.class.getName())));
			
		addSupportedType(new ModuleTypeDTO("MaxPooling" , "Layer", false, 
				new ModulePropertyDTO("Width", "width", Integer.class.getName()),
				new ModulePropertyDTO("Height", "height", Integer.class.getName()),
				new ModulePropertyDTO("Stride X", "strideX", Integer.class.getName()),
				new ModulePropertyDTO("Stride Y", "strideY", Integer.class.getName())));	
		
		addSupportedType(new ModuleTypeDTO("Normalization", "Preprocessing", true));
		
		addSupportedType(new ModuleTypeDTO("Narrow", "Preprocessing", false, 
				new ModulePropertyDTO("Index dim 0", "index0", Integer.class.getName()),
				new ModulePropertyDTO("Size dim 0", "size0", Integer.class.getName()),
				new ModulePropertyDTO("Index dim 1", "index1", Integer.class.getName()),
				new ModulePropertyDTO("Size dim 1", "size1", Integer.class.getName()),
				new ModulePropertyDTO("Index dim 2", "index2", Integer.class.getName()),
				new ModulePropertyDTO("Size dim 2", "size2", Integer.class.getName())));
		
		addSupportedType(new ModuleTypeDTO("Scale", "Preprocessing", false, 
				new ModulePropertyDTO("Dim 0", "dim0", Integer.class.getName()),
				new ModulePropertyDTO("Dim 1", "dim1", Integer.class.getName()),
				new ModulePropertyDTO("Dim 2", "dim2", Integer.class.getName())));
		
		addSupportedType(new ModuleTypeDTO("Frame", "Preprocessing", false, 
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
				module = new Linear(factory, id, parameters, input, output);
			} else {
				module = new Linear(factory, id, input, output);
			}
			break;
		}
		case "Dropout":
		{
			float dropout = Float.parseFloat(dto.properties.get("dropout"));
			module = new Dropout(factory, id, dropout);
			break;
		}
		case "Tanh":
		{
			module = new Tanh(factory, id);
			break;
		}
		case "Sigmoid":
		{
			module = new Sigmoid(factory, id);
			break;
		}
		case "Softmax":
		{
			module = new Softmax(factory, id);
			break;
		}
		case "ReLU":
		{
			module = new ReLU(factory, id);
			break;
		}
		case "PReLU":
		{
			module = new PReLU(factory, id);
			if(parameters!=null){
				((PReLU)module).setParameters(parameters);
			}
			break;
		}
		case "Threshold":
		{
			float thresh = Float.parseFloat(dto.properties.get("thresh"));
			float val = Float.parseFloat(dto.properties.get("val"));
			
			module = new Threshold(factory, id, thresh, val);
			break;
		}
		case "Input":
		{
			module = new InputImpl(factory, id);
			break;
		}
		case "Output":
		{
			module = new OutputImpl(factory, id);
			break;
		}
		case "Duplicate":
		{
			module = new Duplicate(factory, id);
			break;
		}
		case "Accumulate":
		{
			module = new Accumulate(factory, id);
			break;
		}
		case "Multiply":
		{
			module = new Multiply(factory, id);
			break;
		}
		case "Split":
		{
			module = new Split(factory, id);
			break;
		}
		case "Concat":
		{
			module = new Concat(factory, id);
			break;
		}
		case "Grid":
		{
			int x = Integer.parseInt(dto.properties.get("x"));
			int y = Integer.parseInt(dto.properties.get("y"));
			
			int strideX = hasProperty(dto.properties,"strideX") ? Integer.parseInt(dto.properties.get("strideX")) : 1;
			int strideY = hasProperty(dto.properties,"strideY") ? Integer.parseInt(dto.properties.get("strideY")) : 1;

			module = new Grid(factory, id, x, y, strideX, strideY);
			break;
		}
		case "Convolution":
		{
			int noInputPlanes = Integer.parseInt(dto.properties.get("noInputPlanes"));
			int noOutputPlanes = Integer.parseInt(dto.properties.get("noOutputPlanes"));
			int kernelWidth = Integer.parseInt(dto.properties.get("kernelWidth"));
			int kernelHeight = Integer.parseInt(dto.properties.get("kernelHeight"));
			
			int strideX = hasProperty(dto.properties,"strideX") ? Integer.parseInt(dto.properties.get("strideX")) : 1;
			int strideY = hasProperty(dto.properties,"strideY") ? Integer.parseInt(dto.properties.get("strideY")) : 1;

			boolean pad = hasProperty(dto.properties,"pad") ? Boolean.parseBoolean(dto.properties.get("pad")) : false;

			if(parameters!=null){
				module = new SpatialConvolution(factory, id, parameters, noInputPlanes, noOutputPlanes, kernelWidth, kernelHeight, strideX, strideY, pad);
			} else {
				module = new SpatialConvolution(factory, id, noInputPlanes, noOutputPlanes, kernelWidth, kernelHeight, strideX, strideY, pad);
			}
			break;
		}
		case "MaxPooling":
		{
			int width = Integer.parseInt(dto.properties.get("width"));
			int height = Integer.parseInt(dto.properties.get("height"));

			int sx = hasProperty(dto.properties, "strideX") ? Integer.parseInt(dto.properties.get("strideX")) : width;
			int sy = hasProperty(dto.properties,"strideY") ? Integer.parseInt(dto.properties.get("strideY")) : height;
			
			module = new SpatialMaxPooling(factory, id, width, height, sx, sy);
			break;
		}
		case "Normalization":
		{
			module = new Normalization(factory, id);
			if(parameters!=null){
				((Normalization)module).setParameters(parameters);
			}
			break;
		}
		case "Narrow":
		{
			int[] ranges;
			
			int index0 = Integer.parseInt(dto.properties.get("index0"));
			int size0 = Integer.parseInt(dto.properties.get("size0"));
			
			int index1 = Integer.parseInt(dto.properties.get("index1"));
			int size1 = Integer.parseInt(dto.properties.get("size1"));
			
			if(hasProperty(dto.properties, "index2")){
				int index2 = Integer.parseInt(dto.properties.get("index2"));
				int size2 = Integer.parseInt(dto.properties.get("size2"));
				
				ranges = new int[]{index0, size0, index1, size1, index2, size2};
			} else {
				ranges = new int[]{index0, size0, index1, size1};
			}
			
			module = new Narrow(factory, id, ranges);
			break;
		}
		case "Scale":
		{
			int[] dims;
			
			int dim0 = Integer.parseInt(dto.properties.get("dim0"));
			int dim1 = Integer.parseInt(dto.properties.get("dim1"));
			
			if(hasProperty(dto.properties, "dim2")){
				int dim2 = Integer.parseInt(dto.properties.get("dim2"));
				
				dims = new int[]{dim0, dim1, dim2};
			} else {
				dims = new int[]{dim0, dim1};
			}
			
			module = new Scale(factory, id, dims);
			break;
		}
		case "Frame":
		{
			int[] dims;
			
			int dim0 = Integer.parseInt(dto.properties.get("dim0"));
			int dim1 = Integer.parseInt(dto.properties.get("dim1"));
			
			if(hasProperty(dto.properties, "dim2")){
				int dim2 = Integer.parseInt(dto.properties.get("dim2"));
				
				dims = new int[]{dim0, dim1, dim2};
			} else {
				dims = new int[]{dim0, dim1};
			}
			
			module = new Frame(factory, id, dims);
			break;
		}
		case "Masked MaxPooling":
		{
			int noInputs = Integer.parseInt(dto.properties.get("noInputs"));
			String masks = dto.properties.get("masks");
			
			module = new MaskedMaxPooling(factory, id, noInputs, masks);
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
	
	@Reference
	void setTensorFactory(TensorFactory f){
		this.factory = f;
	}
}
