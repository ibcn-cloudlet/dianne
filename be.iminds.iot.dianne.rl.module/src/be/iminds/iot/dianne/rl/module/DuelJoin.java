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
package be.iminds.iot.dianne.rl.module;

import java.util.Iterator;
import java.util.Map.Entry;
import java.util.UUID;

import be.iminds.iot.dianne.api.nn.module.Join;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;

/**
 * Implements the join for a Dueling architecture where one input gives V(s), another input
 * gives A(s, a) and the output will be Q(s, a) as described in
 * 
 * https://arxiv.org/pdf/1511.06581.pdf
 * 
 * @author tverbele
 *
 */
public class DuelJoin extends Join {

	private UUID valueId;
	private UUID advId;
	
	public DuelJoin() {
		super();
	}
	
	public DuelJoin(UUID id) {
		super(id);
	}
	
	@Override
	protected void forward() {
		if(inputs.size() != 2){
			throw new RuntimeException("Wrong number of inputs, 2 expected");
		}
	
		// figure out which input is value and which is adv ... value should have one output only (but might be batched)
		Tensor value = null;
		Tensor adv = null;
		
		Iterator<Entry<UUID, Tensor>> it = inputs.entrySet().iterator();
		Entry<UUID, Tensor> e1 = it.next();
		if(e1.getValue().size(e1.getValue().dim()-1) == 1){
			value = e1.getValue();
			valueId = e1.getKey();
			
			Entry<UUID, Tensor> e2 = it.next();
			adv = e2.getValue();
			advId = e2.getKey();
		} else {
			adv = e1.getValue();
			advId = e1.getKey();
			
			Entry<UUID, Tensor> e2 = it.next();
			value = e2.getValue();
			valueId = e2.getKey();
		}
		
		// construct output
		if(output==null){
			output = new Tensor(adv.dims());
		}
		
		adv.copyInto(output);
		// TODO find something better for this, e.g. expand value tensor?
		if(value.size(0) > 1){
			// batched
			for(int b = 0; b < value.size(0); b++){
				Tensor select = output.select(0, b);
				TensorOps.sub(select, select, TensorOps.mean(adv.select(0, b)));
				TensorOps.add(select, select, value.get(b, 0));
			}
		} else {
			TensorOps.sub(output, output, TensorOps.mean(adv));
			TensorOps.add(output, output, value.get(0));
		}
		
	}

	@Override
	protected void backward() {
		// value gradients
		// valueGrad = sum(gradOutput)
		Tensor valueGrad = gradInputs.get(valueId);
		if(valueGrad == null){
			valueGrad = new Tensor(inputs.get(valueId).dims());
			gradInputs.put(valueId, valueGrad);
		}
		
		if(valueGrad.size(0) > 1){
			// batched
			for(int b = 0; b < valueGrad.size(0); b++){
				valueGrad.set(TensorOps.sum(gradOutput.select(0, b)), b, 0);
			}
		} else {
			valueGrad.set(TensorOps.sum(gradOutput), 0);
		}
		
		
		// advantage gradients
		// advGrad = gradOutput - gradOutput / #actions
		Tensor advGrad = gradInputs.get(advId);
		advGrad = gradOutput.copyInto(advGrad);
		advGrad = TensorOps.div(advGrad, advGrad, -advGrad.size(advGrad.dim()-1));
		advGrad = TensorOps.add(advGrad, advGrad, gradOutput);
		
		gradInputs.put(advId, advGrad);
	}

}
