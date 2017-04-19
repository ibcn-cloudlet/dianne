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
package be.iminds.iot.dianne.nn.learn.processors;

import java.util.HashMap;
import java.util.Map;
import java.util.UUID;

import be.iminds.iot.dianne.api.nn.NeuralNetwork;
import be.iminds.iot.dianne.api.nn.learn.GradientProcessor;
import be.iminds.iot.dianne.nn.learn.processors.config.AdamConfig;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;

public class AdamProcessor extends GradientProcessor {

	private final AdamConfig config;
	
	private final Map<UUID, Tensor> mts = new HashMap<>();
	private final Map<UUID, Tensor> vts = new HashMap<>();

	private Tensor squared = null;
	private Tensor root_v = null;
	
	public AdamProcessor( NeuralNetwork nn, AdamConfig config) {
		super(nn);
		
		this.config = config;
	}
	
	@Override
	public void updateDelta(long i) {
		nn.getTrainables().entrySet().stream().forEach(e -> {
			Tensor deltaParams = e.getValue().getDeltaParameters();
			
			// update biased first momentum estimate
			Tensor mt = mts.get(e.getKey());
			if(mt == null){
				mt = new Tensor(deltaParams.dims());
				mts.put(e.getKey(), mt);
				mt.fill(0.0f);
			}
			mt = TensorOps.mul(mt, mt, config.beta1);
			mt = TensorOps.add(mt, mt, 1-config.beta1, deltaParams);
			
			// update biased second raw momentum estimate
			squared = deltaParams.copyInto(squared);
			squared = TensorOps.cmul(squared, squared, squared);
			
			Tensor vt = vts.get(e.getKey());
			if(vt == null){
				vt = new Tensor(deltaParams.dims());
				vts.put(e.getKey(), vt);
				vt.fill(0.0f);
			}
			vt = TensorOps.mul(vt, vt, config.beta2);
			vt = TensorOps.add(vt, vt, 1-config.beta2, squared);	
			
			
			// update delta params
			float beta1_t = (float) Math.pow(config.beta1, i+1);
			float beta2_t = (float) Math.pow(config.beta2, i+1);

			float at = (float) (config.learningRate*Math.sqrt(1-beta2_t)/(1-beta1_t));
			
			root_v = TensorOps.sqrt(root_v, vt);
			root_v = TensorOps.add(root_v, root_v, config.epsilon);
			deltaParams = TensorOps.cdiv(deltaParams, mt, root_v);
			deltaParams = TensorOps.mul(deltaParams, deltaParams, -at);
			
			// set DeltaParameters to be sure in case of remote module instance
			e.getValue().setDeltaParameters(deltaParams);
		});
	}

}
