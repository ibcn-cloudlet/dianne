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
package be.iminds.iot.dianne.nn.module.preprocessing;

import java.util.UUID;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.nn.module.AbstractModule;
import be.iminds.iot.dianne.api.nn.module.Preprocessor;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;

/**
 * The Normalization module will calculate mean and std over all output values of the dataset
 * before training starts, and do the inverse normalization operations using these parameters
 * at train/runtime.
 * 
 * @author tverbele
 *
 */
public class Denormalization extends AbstractModule implements Preprocessor {

	// TODO per channel normalization?
	private float mean = 0.0f;
	private float std = 1.0f;
	
	private boolean preprocessed = false;
	
	public Denormalization(){
		super();
	}
	
	public Denormalization(UUID id){
		super(id);
	}
	
	public Denormalization(UUID id, float mean, float std){
		super(id);
		this.mean = mean;
		this.std = std;
		preprocessed = true;
	}

	@Override
	protected void forward() {
		output = TensorOps.mul(output, input, std);
		output = TensorOps.add(output, output, mean);
	}

	@Override
	protected void backward() {
		gradInput = TensorOps.div(gradInput, gradOutput, std);
	}

	@Override
	public void preprocess(Dataset data) {
		int n = 0;
		float m = 0;
		float m2 = 0;
		float d,x;
		for(int i=0;i<data.size();i++){
			Tensor output = data.getOutputSample(i);
			float[] outputData = output.get();
			// TODO normalize over all data, what if this is subtensor?
			for(int k=0;k<outputData.length;k++){
				x = outputData[k];
				n++;
				d = x - m;
				m = m + d/n;
				m2 = m2 + d*(x - m);
			}
		}
		
		this.mean = m;
		if(n < 2){
			this.std = 0;
		} else {
			std = (float) Math.sqrt(m2/(n-1));
		}
		
		preprocessed = true;
	}

	@Override
	public Tensor getParameters() {
		Tensor params = new Tensor(2);
		params.set(mean, 0);
		params.set(std, 1);
		return params;
	}

	@Override
	public void setParameters(Tensor params) {
		mean = params.get(0);
		std = params.get(1);
		
		preprocessed = true;
	}

	@Override
	public boolean isPreprocessed() {
		return preprocessed;
	}
}
