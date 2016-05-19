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
package be.iminds.iot.dianne.nn.test.benchmark;

import be.iminds.iot.dianne.api.nn.NeuralNetwork;
import be.iminds.iot.dianne.nn.test.DianneTest;
import be.iminds.iot.dianne.tensor.Tensor;

public class DianneBenchmark extends DianneTest {

	protected long benchmark(String nnName, int[] inputDims, int times, boolean backward) throws Exception {
		System.gc();
		
		NeuralNetwork nn = deployNN(nnName);
		final Tensor input = new Tensor(inputDims);
		input.rand();
		
		// dry run
		for(int i=0;i<10;i++){
			run(nn, input, backward);
		}
		
		long t1 = System.currentTimeMillis();
		for(int i=0;i<times;i++){
			run(nn, input, backward);
		}
		long t2 = System.currentTimeMillis();
		
		return (t2-t1)/times;
	}
	
	private void run(NeuralNetwork nn, Tensor input, boolean backward) throws Exception {
		nn.forward(null, null, input).then(
				p -> {
					Tensor out = p.getValue().tensor;
					// TODO also evaluate criterion here?
					out.rand();
					if(backward)
						return nn.backward(null, null, out);
					else 
						return p;
				}).then(
				p -> {	
					// acc grad
					if(backward)
						nn.getTrainables().values().stream().forEach(m -> m.accGradParameters());
					return p;
				}).getValue();
	}
}
