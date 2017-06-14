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
package be.iminds.iot.dianne.api.dataset;

import java.lang.reflect.Field;
import java.util.Map;

import org.osgi.service.component.annotations.Activate;

import be.iminds.iot.dianne.tensor.Tensor;

public abstract class AbstractDataset implements Dataset {
	
	protected String name;
	protected int[] inputDims;
	protected int inputSize;
	protected String inputType;
	protected int[] targetDims;
	protected int targetSize;
	protected String targetType;
	protected int noSamples;
	protected String[] labels;
	protected String labelsFile;
	protected String dir;
	protected Map<String, Object> properties;

	@Activate
	protected void activate(Map<String, Object> properties) {
		this.properties = properties;
		
		String d = (String)properties.get("dir");
		if(d != null){
			dir = d;
		}

		if(properties.containsKey("name"))
			this.name = (String)properties.get("name");

		String[] id = (String[])properties.get("inputDims");
		if(id!=null){
			inputDims= new int[id.length];
			for(int i=0;i<id.length;i++){
				inputDims[i] = Integer.parseInt(id[i].trim());
			}
		}
		
		if(inputType == null)
			inputType = (String)properties.get("inputType");
		
		String[] od = (String[])properties.get("targetDims");
		if(od != null){
			targetDims= new int[od.length];
			for(int i=0;i<od.length;i++){
				targetDims[i] = Integer.parseInt(od[i].trim());
			}
		}
		
		if(targetType == null)
			targetType = (String)properties.get("targetType");

		
		String ns = (String)properties.get("noSamples");
		if(ns != null)
			noSamples = Integer.parseInt(ns.trim());
		
		if(properties.containsKey("labels")){
			labels = (String[])properties.get("labels");
		} else if(properties.containsKey("labelsFile")){
			labelsFile = (String) properties.get("labelsFile");
		}
		
		init(properties);

		if(labelsFile != null)
			readLabels(labelsFile);
		
		if(inputDims != null){
			inputSize = 1;
			for(int i=0;i<inputDims.length;i++){
				inputSize *= inputDims[i];
			}
		} else {
			inputSize = -1;
		}
		
		if(targetDims != null){
			targetSize = 1;
			for(int i=0;i<targetDims.length;i++){
				targetSize *= targetDims[i];
			}
		} else {
			targetSize = -1;
		}
		
	}
	
	protected abstract void init(Map<String, Object> properties);
	
	protected abstract void readLabels(String labelsFile);

	protected abstract Tensor getInputSample(Tensor t, int index);

	protected abstract Tensor getTargetSample(Tensor t, int index);

	@Override
	public DatasetDTO getDTO(){
		DatasetDTO dto = new DatasetDTO();
		dto.name = getName();
		dto.inputDims = inputDims();
		dto.inputType = inputType();
		dto.targetDims = targetDims();
		dto.targetType = targetType();
		dto.size = size();
		dto.labels = getLabels();
		
		properties.entrySet().forEach(e -> {
			if(e.getKey().contains("."))
				return;
			
			for(Field f : DatasetDTO.class.getFields()){
				if(f.getName().equals(e.getKey()))
					return;
			}
			dto.properties.put(e.getKey(), e.getValue().toString());
		});
		
		return dto;
	}
	
	@Override
	public Sample getSample(Sample s, final int index){
		if(s == null){
			s = new Sample();
		}
		
		s.input = getInputSample(s.input, index);
		s.target = getTargetSample(s.target, index);
		
		return s;
	}
	
	@Override
	public String getName(){
		return name;
	}
	
	@Override
	public int[] inputDims(){
		return inputDims;
	}

	@Override
	public String inputType(){
		return inputType;
	}
	
	@Override
	public int[] targetDims(){
		return targetDims;
	}
	
	@Override
	public String targetType(){
		return targetType;
	}
	
	@Override
	public int size() {
		return noSamples;
	}

	@Override
	public String[] getLabels(){
		return labels;
	}
	
}