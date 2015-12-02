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
package be.iminds.iot.dianne.nn.module.io;

import java.util.UUID;

import be.iminds.iot.dianne.api.nn.module.AbstractModule;
import be.iminds.iot.dianne.api.nn.module.Module;
import be.iminds.iot.dianne.api.nn.module.Output;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;

public class OutputImpl extends AbstractModule implements Output {

	protected String[] labels;
	
	public OutputImpl(TensorFactory factory) {
		super(factory);
	}

	public OutputImpl(TensorFactory factory, UUID id) {
		super(factory, id);
	}
	
	@Override
	public Tensor getOutput(){
		return output;
	}
	
	@Override
	public String[] getTags(){
		return tags;
	}
	
	@Override
	public void backpropagate(final Tensor gradOutput, final String... tags) {
		backward(this.id, gradOutput, tags);
	}
	
	@Override
	protected void forward() {
		output = input;
	}

	@Override
	protected void backward() {
		gradInput = gradOutput;
	}
	
	@Override
	public void setNext(final Module... next) {
		System.out.println("Output cannot have next modules");
	}

	@Override
	public String[] getOutputLabels() {
		return labels;
	}

	@Override
	public void setOutputLabels(String[] labels) {
		this.labels = labels;
	}
}
