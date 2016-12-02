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

public class ELU extends AbstractModule {
	
	private final float alpha;
	private final boolean inPlace = false;

	public ELU() {
		this(1);
	}
	
	public ELU(UUID id) {
		this(id, 1);
	}
	
	public ELU(float alpha) {
		super();
		this.alpha = alpha;
	}
	
	public ELU(UUID id, float beta) {
		super(id);
		this.alpha = beta;
	}
	
	@Override
	protected void forward() {
		output = ModuleOps.elu(output, input, alpha, inPlace);
	}

	@Override
	protected void backward() {
		gradInput = ModuleOps.eluGradIn(gradInput, gradOutput, input, output, alpha, inPlace);
	}

}
