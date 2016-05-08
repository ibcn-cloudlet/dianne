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
import be.iminds.iot.dianne.tensor.ModuleOps;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;

public class PReLU extends AbstractTrainableModule{
	
	private float init = 0.25f;
	
	public PReLU() {
		this(0.25f);
	}
	
	public PReLU(float init) {
		super(new Tensor(1));
		this.init = init;
		init();
	}
	
	public PReLU(UUID id) {
		this(id, 0.25f);
	}

	public PReLU(UUID id, float init) {
		super(id, new Tensor(1));
		this.init = init;
		init();
	}
	
	private void init() {
		parameters.fill(init);
	}

	@Override
	public void randomize(){
		// TODO: write test + check if this is correct
		parameters.set(init, 0);
	}
	
	@Override
	protected void forward() {
		output = ModuleOps.prelu(output, input, parameters, 0); 
	}

	@Override
	protected void backward() {
		if(deltaParameters==null){
			initDeltaParameters(null);
		}
		
		gradInput = ModuleOps.preluGradIn(gradInput, gradOutput, input, parameters, 0);
	}

	private Tensor temp;
	
	@Override
	public void accGradParameters() {
		ModuleOps.preluAccGrad(deltaParameters, gradOutput, input, parameters, 0);
	}

	@Override
	public void initDeltaParameters(Tensor deltas) {
		if(deltas==null){
			deltaParameters = new Tensor(1);
		} else {
			deltaParameters = deltas;
		}
		deltaParameters.fill(0.0f);
	}
}
