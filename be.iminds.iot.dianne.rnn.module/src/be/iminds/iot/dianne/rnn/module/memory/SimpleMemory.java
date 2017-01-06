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
package be.iminds.iot.dianne.rnn.module.memory;

import java.util.UUID;

import be.iminds.iot.dianne.tensor.Tensor;

public class SimpleMemory extends AbstractMemory {

	public SimpleMemory(int size) {
		super(size);
	}

	public SimpleMemory(UUID id, int size) {
		super(id, size);
	}
	
	public SimpleMemory(Tensor t) {
		super(t);
	}

	public SimpleMemory(UUID id, Tensor t) {
		super(id, t);
	}

	@Override
	protected void updateMemory() {
		// simple memory just forwards ...
		memory = input.copyInto(memory);
	}

	@Override
	protected void updateOutput() {
		output = memory.copyInto(output);
	}

	@Override
	protected void resetMemory(int batchSize){
		memory.fill(0.0f);
	}
	
	@Override
	protected void backward() {
		// just forward gradOutput
		gradInput = gradOutput;
	}

}
