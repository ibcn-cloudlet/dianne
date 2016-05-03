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
package be.iminds.iot.dianne.api.nn.module;

import java.util.UUID;

import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;

/**
 * Provides base functionality for trainable neural network Modules. Extend this class
 * for creating your own trainable module with one previous and one next Module.
 * 
 * @author tverbele
 *
 */
public abstract class AbstractTrainableModule extends AbstractModule implements Trainable {

	protected final Tensor parameters;
	protected Tensor deltaParameters;
	protected boolean fixed = false;
	
	public AbstractTrainableModule(Tensor parameters) {
		this.parameters = parameters;
	}
	
	public AbstractTrainableModule(UUID id, Tensor parameters) {
		super(id);
		this.parameters = parameters;
	}
	
	@Override
	public abstract void accGradParameters();

	@Override
	public void zeroDeltaParameters() {
		if(deltaParameters!=null){
			deltaParameters.fill(0.0f);
		}
	}

	@Override
	public void updateParameters() {
		if(!fixed){
			TensorOps.add(parameters, parameters, deltaParameters);
		}
	}
	
	@Override
	public void updateParameters(float scale) {
		if(!fixed){
			TensorOps.add(parameters, parameters, scale, deltaParameters);
		}
	}

	public abstract void initDeltaParameters(Tensor deltas);
	
	@Override
	public Tensor getDeltaParameters(){
		return deltaParameters;
	}
	
	@Override
	public void setDeltaParameters(Tensor deltas){
		deltas.copyInto(deltaParameters);
	}
	
	@Override
	public Tensor getParameters(){
		return parameters;
	}
	
	@Override
	public void setParameters(Tensor params){
		params.copyInto(parameters);
	}
}
