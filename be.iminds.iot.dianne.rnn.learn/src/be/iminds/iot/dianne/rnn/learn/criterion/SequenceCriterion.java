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
package be.iminds.iot.dianne.rnn.learn.criterion;

import java.util.ArrayList;
import java.util.List;

import be.iminds.iot.dianne.api.nn.learn.Criterion;
import be.iminds.iot.dianne.rnn.learn.criterion.SequenceCriterionFactory.SequenceCriterionConfig;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;

/**
 * Wrapper for processing a Criterion on a Tensor sequence
 * @author tverbele
 *
 */
public class SequenceCriterion implements Criterion {

	private Criterion criterion;
	private SequenceCriterionConfig config;
	
	private List<Tensor> losses = new ArrayList<>();
	private List<Tensor> grads = new ArrayList<>();
	
	public SequenceCriterion(Criterion c, SequenceCriterionConfig conf) {
		this.criterion = c;
		this.config = conf;
	}
	
	@Override
	public Tensor loss(final Tensor output, final Tensor target) {
		return criterion.loss(output, target);
	}

	@Override
	public Tensor grad(final Tensor output, final Tensor target) {
		return criterion.grad(output, target);
	}
	
	public List<Tensor> loss(final List<Tensor> outputs, final List<Tensor> targets){
		// Criterion expects to call grad immediately after loss, so we do both here
		// and just return latest grads array in grad call
		for(int i=0;i<outputs.size();i++){
			Tensor loss = criterion.loss(outputs.get(i), targets.get(i)).clone();
			Tensor grad = criterion.grad(outputs.get(i), targets.get(i));

			if(!config.backpropAll && i!=outputs.size()-1){
				loss.fill(0.0f);
				grad.fill(0.0f);
			}
			if(losses.size() <= i){
				losses.add(loss.clone());
				grads.add(grad.clone());
			} else {
				loss.copyInto(losses.get(i));
				grad.copyInto(grads.get(i));
			}
		}
		return losses;
	}
	
	public List<Tensor> grad(final List<Tensor> outputs, final List<Tensor> targets){
		// grads list should be updated with a loss call
		return grads;
	}
}
