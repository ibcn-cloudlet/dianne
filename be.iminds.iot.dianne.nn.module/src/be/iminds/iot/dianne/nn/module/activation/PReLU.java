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
 *     Tim Verbelen, Steven Bohez, Elias De Coninck
 *******************************************************************************/
package be.iminds.iot.dianne.nn.module.activation;

import java.util.UUID;

import be.iminds.iot.dianne.api.nn.module.AbstractTrainableModule;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;

public class PReLU extends AbstractTrainableModule{
	
	private float init = 0.25f;
	
	public PReLU(TensorFactory factory) {
		this(factory, 0.25f);
	}
	
	public PReLU(TensorFactory factory, float init) {
		super(factory, factory.createTensor(1));
		this.init = init;
		init();
	}
	
	public PReLU(TensorFactory factory, UUID id) {
		this(factory, id, 0.25f);
	}

	public PReLU(TensorFactory factory, UUID id, float init) {
		super(factory, id, factory.createTensor(1));
		this.init = init;
		init();
	}
	
	private void init() {
		parameters.fill(0.0f);
	}

	@Override
	public void randomize(){
		parameters.set(init, 0);
	}
	
	@Override
	protected void forward() {
		output = factory.getTensorMath().thresh(output, input, 0f, parameters.get(0), 0f);
	}

	@Override
	protected void backward() {
		if(deltaParameters==null){
			initDeltaParameters(null);
		}
		gradInput = factory.getTensorMath().cmul(gradInput, gradOutput,
				factory.getTensorMath().dthresh(gradInput, input, 0f, parameters.get(0)));
	}

	private Tensor temp;
	
	@Override
	public void accGradParameters() {
		temp = factory.getTensorMath().mul(temp, input, -1f);
		temp = factory.getTensorMath().thresh(temp, temp, 0f, 0f, 0f);
		temp = factory.getTensorMath().mul(temp, temp, -1f);
		
		deltaParameters = factory.getTensorMath().add(deltaParameters, deltaParameters,
				factory.getTensorMath().dot(temp, gradOutput));
	}

	@Override
	public void initDeltaParameters(Tensor deltas) {
		if(deltas==null){
			deltaParameters = factory.createTensor(1);
		} else {
			deltaParameters = deltas;
		}
		deltaParameters.fill(0.0f);
	}
}
