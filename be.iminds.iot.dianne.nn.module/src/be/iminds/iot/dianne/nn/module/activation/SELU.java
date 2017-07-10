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
package be.iminds.iot.dianne.nn.module.activation;

import java.util.UUID;

import be.iminds.iot.dianne.api.nn.module.AbstractModule;
import be.iminds.iot.dianne.tensor.ModuleOps;

/**
 * 
 * Activation according to the Self-Normalizing Neural Networks paper
 * 
 * https://arxiv.org/pdf/1706.02515.pdf
 * 
 * @author tverbele
 *
 */
public class SELU extends AbstractModule {

	// these values are fixed from the paper... no need to make configurable?
	private final float lambda = 1.0507f;
	private final float alpha = 1.6733f;
	
	public SELU() {
		super();
	}

	public SELU(UUID id) {
		super(id);
	}

	@Override
	protected void forward() {
		output = ModuleOps.selu(output, input, alpha, lambda);
	}

	@Override
	protected void backward() {
		gradInput = ModuleOps.seluGradIn(gradInput, gradOutput, input, output, alpha, lambda);
	}
	
}
